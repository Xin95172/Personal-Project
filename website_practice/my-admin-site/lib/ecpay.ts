import crypto from "crypto";

function ecpayEncode(value: string) {
  return encodeURIComponent(value)
    .replace(/%20/g, "+")
    .replace(/%21/g, "!")
    .replace(/%28/g, "(")
    .replace(/%29/g, ")")
    .replace(/%2A/g, "*")
    .replace(/%2D/g, "-")
    .replace(/%2E/g, ".")
    .replace(/%5F/g, "_")
    .toLowerCase();
}

export function generateCheckMacValue(
  params: Record<string, string>,
  hashKey: string,
  hashIV: string
) {
  const sorted = Object.keys(params)
    .filter((key) => key !== "CheckMacValue")
    .sort((a, b) => a.localeCompare(b))
    .map((key) => `${key}=${params[key]}`)
    .join("&");

  const raw = `HashKey=${hashKey}&${sorted}&HashIV=${hashIV}`;

  const encoded = ecpayEncode(raw);

  return crypto
    .createHash("sha256")
    .update(encoded)
    .digest("hex")
    .toUpperCase();
}