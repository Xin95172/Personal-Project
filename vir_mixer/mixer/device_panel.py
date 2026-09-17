"""Nonblocking endpoint scanning and searchable input/output inventory."""
import json
import sys
from pathlib import Path
from PySide6.QtCore import Qt, QProcess, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)


class DeviceScanner(QObject):
    updated = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.error)
        self.timeout = QTimer(self)
        self.timeout.setSingleShot(True)
        self.timeout.timeout.connect(self.timed_out)
        self.timer = QTimer(self)
        self.timer.setInterval(5000)
        self.timer.timeout.connect(self.scan)
        self.stopping = False

    def scan(self):
        if not self.stopping and self.process.state() == QProcess.ProcessState.NotRunning:
            self.process.start(sys.executable, ['-X', 'utf8', '-m', 'mixer.devices'])
            self.timeout.start(10000)

    def finished(self, code, status):
        self.timeout.stop()
        if self.stopping:
            return
        if code != 0:
            self.failed.emit('装置掃描失敗：' + bytes(self.process.readAllStandardError()).decode('utf-8', errors='replace')[-500:])
            return
        try:
            data = json.loads(bytes(self.process.readAllStandardOutput()).decode('utf-8'))
            self.updated.emit(data)
        except Exception as exc:
            self.failed.emit(f'無法讀取裝置清單：{exc}')

    def error(self, error):
        if not self.stopping:
            self.failed.emit(f'無法啟動裝置掃描：{self.process.errorString()}')

    def timed_out(self):
        self.process.kill()
        if not self.stopping:
            self.failed.emit('裝置掃描逾時，請稍後重試。')

    def stop(self):
        self.stopping = True
        self.timer.stop()
        self.timeout.stop()
        self.process.kill()


class DevicePanel(QDialog):
    output_selected = Signal(object)
    scan_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('音訊裝置 · 所有輸入與輸出')
        self.resize(1000, 580)
        self.data = []
        self.output_active = False
        layout = QVBoxLayout(self)
        note = QLabel('列出音訊系統可偵測的所有端點。同一硬體可能以 WASAPI、MME 等不同介面出現。\n輸入裝置在此供檢視；目前音軌來源仍為音檔，不會自動開啟麥克風。')
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText('搜尋裝置名稱或音訊介面…')
        self.search.textChanged.connect(self.populate)
        row.addWidget(self.search, 1)
        self.direction = QComboBox()
        self.direction.addItems(['全部', '輸入', '輸出'])
        self.direction.currentIndexChanged.connect(self.populate)
        row.addWidget(self.direction)
        scan = QPushButton('立即重新掃描')
        scan.clicked.connect(self.scan_requested.emit)
        row.addWidget(scan)
        layout.addLayout(row)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['裝置名稱', '方向', '輸入聲道', '輸出聲道', '預設取樣率', '音訊介面', '系統預設'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        layout.addWidget(self.table, 1)
        self.use = QPushButton('設為主輸出')
        self.use.clicked.connect(self.choose)
        self.use.setEnabled(False)
        layout.addWidget(self.use)

    def set_inventory(self, data):
        self.data = data
        self.populate()

    def populate(self, *_):
        selected = self.selected()
        query = self.search.text().casefold()
        direction = self.direction.currentIndex()
        rows = [d for d in self.data if query in (d['name'] + d['host']).casefold()
                and (direction != 1 or d['inputs']) and (direction != 2 or d['outputs'])]
        self.table.setRowCount(0)
        for row, d in enumerate(rows):
            self.table.insertRow(row)
            kind = '輸入 / 輸出' if d['inputs'] and d['outputs'] else '輸入' if d['inputs'] else '輸出'
            default = ' / '.join(name for name, enabled in [('輸入', d['default_input']), ('輸出', d['default_output'])] if enabled)
            values = [d['name'], kind, d['inputs'], d['outputs'], f"{d['sample_rate']:g} Hz", d['host'], default]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, d)
                self.table.setItem(row, col, item)
            if selected and d['name'] == selected['name'] and d['host'] == selected['host']:
                self.table.selectRow(row)
        self.summary.setText(f"偵測到 {sum(d['inputs'] > 0 for d in self.data)} 個輸入端點、{sum(d['outputs'] > 0 for d in self.data)} 個輸出端點 · 顯示 {len(rows)} 列")
        self.selection_changed()

    def selected(self):
        item = self.table.item(self.table.currentRow(), 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def selection_changed(self):
        selected = self.selected()
        self.use.setEnabled(bool(selected and selected['outputs']) and not self.output_active)
        self.use.setToolTip('切換裝置前請先關閉輸出' if self.output_active else '')

    def choose(self):
        if self.use.isEnabled():
            self.output_selected.emit(self.selected())
