import { ApiRequestError } from "../api/client";

export interface ErrorPresentation {
  title: string;
  message: string;
  category: "validation" | "readiness" | "backpressure" | "timeout" | "server";
}

const validationCodes = new Set([
  "invalid_request",
  "empty_upload",
  "unsupported_media_type",
  "upload_too_large",
  "invalid_image",
  "image_dimensions_exceeded",
]);

export function presentRequestError(cause: unknown): ErrorPresentation {
  if (cause instanceof ApiRequestError) {
    if (validationCodes.has(cause.code)) {
      return { title: "Image rejected", message: cause.message, category: "validation" };
    }
    if (cause.code === "model_unavailable") {
      return { title: "Model unavailable", message: cause.message, category: "readiness" };
    }
    if (cause.code === "inference_busy") {
      return {
        title: "Service is busy",
        message: "The inference queue is full. Keep the image selected and try again shortly.",
        category: "backpressure",
      };
    }
    if (cause.code === "inference_timeout") {
      return {
        title: "Processing timed out",
        message: "The operation exceeded the service time limit. The selected image was retained.",
        category: "timeout",
      };
    }
    if (cause.code === "offline") {
      return { title: "API offline", message: cause.message, category: "readiness" };
    }
  }
  return {
    title: "Request failed safely",
    message: "The operation could not be completed. The selected image was retained.",
    category: "server",
  };
}
