export class RestoredImagePayloadError extends Error {
  constructor() {
    super("The restored image payload is invalid.");
    this.name = "RestoredImagePayloadError";
  }
}

export function restoredPngBlob(content: string): Blob {
  try {
    const binary = atob(content);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
    if (bytes.length < signature.length || signature.some((value, index) => bytes[index] !== value)) {
      throw new RestoredImagePayloadError();
    }
    return new Blob([bytes], { type: "image/png" });
  } catch (cause) {
    if (cause instanceof RestoredImagePayloadError) throw cause;
    throw new RestoredImagePayloadError();
  }
}

export function restoredDownloadName(originalName: string): string {
  const stem = originalName.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "_");
  return `${stem || "sem-image"}-restored.png`;
}
