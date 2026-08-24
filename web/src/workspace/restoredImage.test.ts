import { describe, expect, it } from "vitest";

import {
  restoredDownloadName,
  RestoredImagePayloadError,
  restoredPngBlob,
} from "./restoredImage";

const pngContent = "iVBORw0KGgoAAAANSUhEUg==";

describe("restored image transport", () => {
  it("decodes the exact PNG bytes into a lossless blob", async () => {
    const blob = restoredPngBlob(pngContent);

    expect(blob.type).toBe("image/png");
    expect(Array.from(new Uint8Array(await blob.arrayBuffer())).slice(0, 8)).toEqual([
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
    ]);
  });

  it("rejects malformed and non-PNG payloads safely", () => {
    expect(() => restoredPngBlob("not base64 ###")).toThrow(RestoredImagePayloadError);
    expect(() => restoredPngBlob(btoa("not a png"))).toThrow(RestoredImagePayloadError);
  });

  it("creates a portable PNG download name", () => {
    expect(restoredDownloadName("wafer field 07.tiff")).toBe("wafer_field_07-restored.png");
    expect(restoredDownloadName(".tiff")).toBe("sem-image-restored.png");
  });
});
