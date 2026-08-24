export const SUPPORTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/tiff"] as const;

export class LocalImageValidationError extends Error {
  readonly code: "empty_upload" | "unsupported_media_type" | "invalid_image";

  constructor(
    code: "empty_upload" | "unsupported_media_type" | "invalid_image",
    message: string,
  ) {
    super(message);
    this.name = "LocalImageValidationError";
    this.code = code;
  }
}

function ensureRange(view: DataView, offset: number, length: number): void {
  if (offset < 0 || length < 0 || offset + length > view.byteLength) {
    throw new LocalImageValidationError("invalid_image", "The image header is incomplete.");
  }
}

function pngDimensions(view: DataView): [number, number] {
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  ensureRange(view, 0, 24);
  if (signature.some((value, index) => view.getUint8(index) !== value)) {
    throw new LocalImageValidationError("invalid_image", "The file is not a valid PNG image.");
  }
  return [view.getUint32(16), view.getUint32(20)];
}

function jpegDimensions(view: DataView): [number, number] {
  ensureRange(view, 0, 4);
  if (view.getUint16(0) !== 0xffd8) {
    throw new LocalImageValidationError("invalid_image", "The file is not a valid JPEG image.");
  }
  let offset = 2;
  while (offset + 4 <= view.byteLength) {
    if (view.getUint8(offset) !== 0xff) {
      offset += 1;
      continue;
    }
    const marker = view.getUint8(offset + 1);
    offset += 2;
    if (marker === 0xd8 || marker === 0xd9 || marker === 0x01) continue;
    ensureRange(view, offset, 2);
    const length = view.getUint16(offset);
    if (length < 2) break;
    const isStartOfFrame =
      (marker >= 0xc0 && marker <= 0xc3) ||
      (marker >= 0xc5 && marker <= 0xc7) ||
      (marker >= 0xc9 && marker <= 0xcb) ||
      (marker >= 0xcd && marker <= 0xcf);
    if (isStartOfFrame) {
      ensureRange(view, offset, 7);
      return [view.getUint16(offset + 5), view.getUint16(offset + 3)];
    }
    offset += length;
  }
  throw new LocalImageValidationError("invalid_image", "JPEG dimensions could not be read.");
}

function tiffValue(
  view: DataView,
  entryOffset: number,
  littleEndian: boolean,
  type: number,
  count: number,
): number | null {
  if (count !== 1 || (type !== 3 && type !== 4)) return null;
  return type === 3
    ? view.getUint16(entryOffset + 8, littleEndian)
    : view.getUint32(entryOffset + 8, littleEndian);
}

function tiffDimensions(view: DataView): [number, number] {
  ensureRange(view, 0, 8);
  const byteOrder = view.getUint16(0);
  const littleEndian = byteOrder === 0x4949;
  if (!littleEndian && byteOrder !== 0x4d4d) {
    throw new LocalImageValidationError("invalid_image", "The file is not a valid TIFF image.");
  }
  if (view.getUint16(2, littleEndian) !== 42) {
    throw new LocalImageValidationError("invalid_image", "The file is not a valid TIFF image.");
  }
  const directoryOffset = view.getUint32(4, littleEndian);
  ensureRange(view, directoryOffset, 2);
  const entryCount = view.getUint16(directoryOffset, littleEndian);
  ensureRange(view, directoryOffset + 2, entryCount * 12);
  let width: number | null = null;
  let height: number | null = null;
  for (let index = 0; index < entryCount; index += 1) {
    const entryOffset = directoryOffset + 2 + index * 12;
    const tag = view.getUint16(entryOffset, littleEndian);
    if (tag !== 256 && tag !== 257) continue;
    const value = tiffValue(
      view,
      entryOffset,
      littleEndian,
      view.getUint16(entryOffset + 2, littleEndian),
      view.getUint32(entryOffset + 4, littleEndian),
    );
    if (tag === 256) width = value;
    if (tag === 257) height = value;
  }
  if (width === null || height === null) {
    throw new LocalImageValidationError("invalid_image", "TIFF dimensions could not be read.");
  }
  return [width, height];
}

export async function inspectImageFile(file: File): Promise<{ width: number; height: number }> {
  if (file.size === 0) {
    throw new LocalImageValidationError("empty_upload", "Select a non-empty image file.");
  }
  if (!SUPPORTED_IMAGE_TYPES.includes(file.type as (typeof SUPPORTED_IMAGE_TYPES)[number])) {
    throw new LocalImageValidationError(
      "unsupported_media_type",
      "Choose a PNG, JPEG, or single-frame TIFF image.",
    );
  }
  const view = new DataView(await file.arrayBuffer());
  const dimensions =
    file.type === "image/png"
      ? pngDimensions(view)
      : file.type === "image/jpeg"
        ? jpegDimensions(view)
        : tiffDimensions(view);
  const [width, height] = dimensions;
  if (width <= 0 || height <= 0) {
    throw new LocalImageValidationError("invalid_image", "The image dimensions are invalid.");
  }
  return { width, height };
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}
