import { NextResponse } from "next/server";

const DETAIL_URL =
  "https://cloud.tipo.gov.tw/S282/S282BV1/api/result/list";

type DocInput =
  | string
  | {
      docid?: string;
    };

export async function POST(req: Request) {
  try {
    const body = await req.json();

    const rawDocs: DocInput[] = Array.isArray(body.docs)
      ? body.docs
      : [];

    // 就算前端不小心傳物件，也自動抽出 docid
    const docs = rawDocs
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }

        return item?.docid;
      })
      .filter(
        (docid): docid is string =>
          typeof docid === "string" && docid.length > 0
      );

    if (docs.length === 0) {
      return NextResponse.json(
        { error: "docs required" },
        { status: 400 }
      );
    }

    console.log("Sending doc IDs:", docs.length);
    console.log("First doc ID:", docs[0]);

    const res = await fetch(DETAIL_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json, text/plain, */*",
        Origin: "https://cloud.tipo.gov.tw",
        Referer:
          "https://cloud.tipo.gov.tw/S282/S282BV1/",
      },
      body: JSON.stringify({
        docs,
      }),
      cache: "no-store",
    });

    const text = await res.text();

    console.log("TIPO detail status:", res.status);

    if (!res.ok) {
      console.error("TIPO detail response:", text);

      return NextResponse.json(
        {
          error: "TIPO detail failed",
          tipoStatus: res.status,
          tipoResponse: text,
        },
        { status: 502 }
      );
    }

    const data = JSON.parse(text);

    return NextResponse.json(data);
  } catch (error) {
    console.error("Trademark detail error:", error);

    return NextResponse.json(
      { error: "server error" },
      { status: 500 }
    );
  }
}