import fs from "fs";
import path from "path";

let cached: Record<string, any> | null | undefined;

export function getPrivateExt(): Record<string, any> | null {
  if (cached !== undefined) return cached;
  try {
    const extPath = path.join(process.cwd(), "src", "private", "sources.cjs");
    if (!fs.existsSync(extPath)) {
      cached = null;
      return null;
    }
    const req = eval("require") as NodeRequire;
    cached = req(extPath);
    return cached;
  } catch {
    cached = null;
    return null;
  }
}
