import { searchIndex } from "@/lib/site";

// A static file, written at build time. The search dialog loads it when the reader opens it.
export const dynamic = "force-static";

export function GET() {
  return Response.json(searchIndex());
}
