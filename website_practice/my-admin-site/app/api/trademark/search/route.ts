import { NextResponse } from "next/server";

const WORD_SEARCH_URL =
  "https://cloud.tipo.gov.tw/S282/S282BV1/api/search/wordSearch";

type TipoSearchDoc = {
  docid?: string;
};

export async function POST(req: Request) {
  try {
    const { keyword } = await req.json();

    const cleanKeyword =
      typeof keyword === "string" ? keyword.trim() : "";

    if (!cleanKeyword) {
      return NextResponse.json(
        { error: "請輸入商標名稱" },
        { status: 400 }
      );
    }

    const tipoResponse = await fetch(WORD_SEARCH_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        tmarkDraft: cleanKeyword,
        records: [
          {
            name: "商標文字",
            value: cleanKeyword,
            oper: "文字近似",
            logic: "AND",
          },
        ],
      }),
      cache: "no-store",
    });

    if (!tipoResponse.ok) {
      const text = await tipoResponse.text();

      console.error(
        "TIPO wordSearch failed:",
        tipoResponse.status,
        text
      );

      return NextResponse.json(
        { error: "TIPO search failed" },
        { status: 502 }
      );
    }

    const data = await tipoResponse.json();

    // 關鍵：只留下真正的 docid 字串
    const docs = Array.isArray(data.docs)
      ? data.docs
          .map((item: string | TipoSearchDoc) => {
            if (typeof item === "string") {
              return item;
            }

            return item?.docid;
          })
          .filter(
            (docid: unknown): docid is string =>
              typeof docid === "string" && docid.length > 0
          )
      : [];

    return NextResponse.json({
      success: true,
      numFound: Number(data.numFound ?? 0),
      docs,
    });
  } catch (error) {
    console.error("Trademark search error:", error);

    return NextResponse.json(
      { error: "server error" },
      { status: 500 }
    );
  }
}