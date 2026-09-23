const CAMERA_CONSTRAINTS: MediaStreamConstraints = {
  audio: false,
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 },
    facingMode: { ideal: 'user' },
  },
}

export async function requestCameraStream(): Promise<MediaStream> {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera access requires localhost or a secure HTTPS page.')
  }
  return navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS)
}

export async function attachCamera(video: HTMLVideoElement, stream: MediaStream): Promise<void> {
  video.srcObject = stream
  video.muted = true
  video.playsInline = true
  await video.play()

  if (video.videoWidth > 0 && video.videoHeight > 0) return

  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(() => finish(new Error('The camera did not provide a video frame.')), 10000)
    const onReady = () => finish()
    const onError = () => finish(new Error('The camera video could not start.'))
    const finish = (error?: Error) => {
      window.clearTimeout(timeout)
      video.removeEventListener('loadedmetadata', onReady)
      video.removeEventListener('error', onError)
      if (error) reject(error)
      else resolve()
    }
    video.addEventListener('loadedmetadata', onReady)
    video.addEventListener('error', onError)
    if (video.videoWidth > 0 && video.videoHeight > 0) finish()
  })
}

export function stopCamera(video: HTMLVideoElement, stream: MediaStream | null): void {
  stream?.getTracks().forEach((track) => track.stop())
  if (video.srcObject === stream) {
    video.pause()
    video.srcObject = null
  }
}

export function cameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    switch (error.name) {
      case 'NotAllowedError':
      case 'PermissionDeniedError':
        return 'Camera permission was denied. Allow camera access in your browser and try again.'
      case 'NotFoundError':
      case 'DevicesNotFoundError':
        return 'No camera was found. Connect a camera and try again.'
      case 'NotReadableError':
      case 'TrackStartError':
        return 'The camera is busy or unavailable. Close other camera apps and try again.'
    }
  }
  return error instanceof Error ? error.message : 'The camera could not start.'
}
