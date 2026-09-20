#define VIRMIXER_DIAGNOSTICS 1
#include "../core/VirtualCable.h"
#include <string.h>

int main(int argc, char** argv) {
    VirtualCable cable;
    cable.Initialize();
    BYTE input[3840], output[3840]; memset(input,42,sizeof(input));
    for(unsigned side=0;side<2;++side) {
        cable.State(side!=0,0,3,false,side+1,0,0);
        cable.RunClock(side!=0,1000000,10000000,1000000,0,0);
        cable.State(side!=0,0,3,true,side+1,0,0);
    }
    auto position=[&](bool capture, ULONG displacement) {
        cable.Position(capture,1200000,1000000,1200000,0,0,0,displacement,7680,20,2,3,3,false);
    };
    position(true,1920); cable.Read(output,1920); // startup silence
    position(false,3840); cable.Write(input,3840);
    position(true,3840); cable.Read(output,3840);
    assert(memcmp(input,output,3840)==0);
    position(true,192); cable.Read(output,192); // primed underrun
    for(unsigned i=0;i<192;++i) assert(output[i]==0);
    cable.Notification(false,1200000,20,true,0,3840,1,1,0,3840,7680,2);
    cable.Notification(false,1200010,20,true,0,3840,2,1,0,3840,7680,2); // no boundary crossed
    cable.Reset(); // between position record and transfer, intentionally
    cable.Write(input,3840);
    cable.FlushStopped(); assert(debugRecords.empty());
    cable.State(false,3,0,true,1,0,0);
    cable.FlushStopped(); assert(debugRecords.empty());
    cable.State(true,3,0,true,2,0,0);
    cable.FlushStopped(); assert(!debugRecords.empty());
    std::string all;
    for(const auto& line:debugRecords) all+=line;
    assert(all.find("W=3840 Req=5952 Actual=3840")!=std::string::npos);
    assert(all.find("prefillBytes=1920 shortageBytes=192")!=std::string::npos);
    assert(all.find("kind=10")!=std::string::npos);
    assert(all.find("signals=1 crossed=0")!=std::string::npos);
    assert(all.find("TRACE_END dropped=0")!=std::string::npos);
    if(argc==2) {
        FILE* file=fopen(argv[1],"wb"); assert(file);
        assert(fwrite(all.data(),1,all.size(),file)==all.size()); fclose(file);
    }
    puts("PASS: diagnostic wrapper preserves PCM/silence accounting, observes epochs and notifications, prints only after both STOP with no lock held (user-mode stubs)");
}
