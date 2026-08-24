import { describe, expect, it } from "vitest";

import { inspectImageFile, LocalImageValidationError } from "./imageFile";

function blobPart(bytes: Uint8Array): ArrayBuffer {
  const copy = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(copy).set(bytes);
  return copy;
}

function png(width: number, height: number): Uint8Array {
  const bytes = new Uint8Array(24);
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const view = new DataView(bytes.buffer);
  view.setUint32(16, width);
  view.setUint32(20, height);
  return bytes;
}

function jpeg(width: number, height: number): Uint8Array {
  return new Uint8Array([
    0xff, 0xd8,
    0xff, 0xc0, 0x00, 0x11, 0x08,
    (height >> 8) & 0xff, height & 0xff,
    (width >> 8) & 0xff, width & 0xff,
    0x01, 0x01, 0x11, 0x00,
    0xff, 0xd9,
  ]);
}

function tiff(width: number, height: number): Uint8Array {
  const bytes = new Uint8Array(38);
  const view = new DataView(bytes.buffer);
  view.setUint16(0, 0x4949);
  view.setUint16(2, 42, true);
  view.setUint32(4, 8, true);
  view.setUint16(8, 2, true);
  view.setUint16(10, 256, true);
  view.setUint16(12, 4, true);
  view.setUint32(14, 1, true);
  view.setUint32(18, width, true);
  view.setUint16(22, 257, true);
  view.setUint16(24, 4, true);
  view.setUint32(26, 1, true);
  view.setUint32(30, height, true);
  return bytes;
}

describe("local image inspection", () => {
  it.each([
    ["sample.png", "image/png", png(17, 9), 17, 9],
    ["sample.jpg", "image/jpeg", jpeg(31, 15), 31, 15],
    ["sample.tiff", "image/tiff", tiff(41, 23), 41, 23],
  ])("reads %s dimensions without decoding or persistence", async (name, type, bytes, width, height) => {
    await expect(inspectImageFile(new File([blobPart(bytes)], name, { type }))).resolves.toEqual({
      width,
      height,
    });
  });

  it("rejects empty, unsupported, and mismatched files safely", async () => {
    await expect(inspectImageFile(new File([], "empty.png", { type: "image/png" }))).rejects.toMatchObject({
      code: "empty_upload",
    });
    await expect(
      inspectImageFile(new File(["gif"], "sample.gif", { type: "image/gif" })),
    ).rejects.toMatchObject({ code: "unsupported_media_type" });
    await expect(
      inspectImageFile(new File([blobPart(jpeg(2, 2))], "wrong.png", { type: "image/png" })),
    ).rejects.toBeInstanceOf(LocalImageValidationError);
  });
});
