const CAMERA_CONSTRAINTS: MediaStreamConstraints = {
  audio: false,
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 },
    facingMode: { ideal: 'user' },
  },
}

/** Process actual presented camera frames, not every change in video.currentTime. */
export class CameraFrameGate {
  private video: HTMLVideoElement | null = null
  private callbackId: number | null = null
  private generation = 0
  private pending = false
  private lastCallbackAt = 0
  private lastQualityCount: number | null = null
  private lastCurrentTime = -1
  private lastProcessedAt = 0

  start(video: HTMLVideoElement, now: number): void {
    this.stop()
    this.video = video
    this.lastCallbackAt = now
    this.lastProcessedAt = now
    this.lastCurrentTime = video.currentTime
    this.lastQualityCount = this.qualityCount(video)
    const generation = this.generation
    if (typeof video.requestVideoFrameCallback !== 'function') return

    const onFrame: VideoFrameRequestCallback = (timeMs) => {
      if (generation !== this.generation || this.video !== video) return
      this.pending = true
      this.lastCallbackAt = timeMs / 1000
      this.callbackId = video.requestVideoFrameCallback(onFrame)
    }
    try {
      this.callbackId = video.requestVideoFrameCallback(onFrame)
    } catch {
      this.callbackId = null
    }
  }

  consume(now: number): boolean {
    const video = this.video
    if (!video) return false
    const count = this.qualityCount(video)
    if (this.pending) {
      this.pending = false
      this.recordFrame(now, count)
      return true
    }
    // Some browsers stop video-frame callbacks for an invisible <video>.
    // A displayed-frame counter is a fallback; currentTime alone is last resort.
    if (this.callbackId !== null && now - this.lastCallbackAt < 0.1) return false
    if (count !== null && (this.lastQualityCount === null || count > this.lastQualityCount)) {
      this.recordFrame(now, count)
      return true
    }
    if ((count === null || count === 0) && now - this.lastProcessedAt >= 1 / 30 &&
        video.currentTime > this.lastCurrentTime) {
      this.recordFrame(now, count)
      return true
    }
    return false
  }

  stop(): void {
    this.generation += 1
    if (this.video && this.callbackId !== null) {
      try { this.video.cancelVideoFrameCallback(this.callbackId) } catch { /* The video may already have stopped. */ }
    }
    this.video = null
    this.callbackId = null
    this.pending = false
    this.lastQualityCount = null
    this.lastCurrentTime = -1
  }

  private recordFrame(now: number, count: number | null): void {
    this.lastProcessedAt = now
    this.lastQualityCount = count
    this.lastCurrentTime = this.video?.currentTime ?? -1
  }

  private qualityCount(video: HTMLVideoElement): number | null {
    try {
      const quality = video.getVideoPlaybackQuality?.()
      const count = quality ? quality.totalVideoFrames - (quality.droppedVideoFrames ?? 0) : undefined
      return count !== undefined && Number.isFinite(count) ? count : null
    } catch { return null }
  }
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
