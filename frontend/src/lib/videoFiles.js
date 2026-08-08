export const MAX_UPLOAD_BYTES = 80 * 1024 * 1024;
export const VIDEO_ACCEPT = 'video/mp4,video/webm,video/quicktime,video/x-msvideo,video/*';
export const SUPPORTED_VIDEO_PATTERN = /\.(webm|mp4|mov|m4v|avi)$/i;

export function formatFileSize(bytes = 0) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function isSupportedVideoFile(file) {
  if (!file) return false;
  return file.type?.startsWith('video/') || SUPPORTED_VIDEO_PATTERN.test(file.name || '');
}
