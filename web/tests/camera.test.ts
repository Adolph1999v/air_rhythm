import { describe, expect, it } from 'vitest'
import { CameraFrameGate } from '../src/camera'

function cameraWithCallbacks() {
  let callback: VideoFrameRequestCallback | null = null
  let frames = 0
  let cancelled = 0
  const video = {
    currentTime: 0,
    requestVideoFrameCallback(next: VideoFrameRequestCallback) { callback = next; return 1 },
    cancelVideoFrameCallback() { cancelled += 1 },
    getVideoPlaybackQuality() { return { totalVideoFrames: frames } },
  } as unknown as HTMLVideoElement
  return {
    video,
    present(timeMs: number) {
      frames += 1
      video.currentTime = frames / 30
      const next = callback
      if (!next) throw new Error('The video callback was not registered.')
      next(timeMs, {} as VideoFrameCallbackMetadata)
    },
    get cancelled() { return cancelled },
  }
}

describe('camera-frame pacing', () => {
  it('does not treat a moving video clock as a new camera image', () => {
    const camera = cameraWithCallbacks()
    const gate = new CameraFrameGate()
    gate.start(camera.video, 1)
    camera.video.currentTime = 0.016
    expect(gate.consume(1.016)).toBe(false)
    camera.present(1033)
    expect(gate.consume(1.033)).toBe(true)
    expect(gate.consume(1.050)).toBe(false)
    gate.stop()
    expect(camera.cancelled).toBe(1)
    expect(gate.consume(1.066)).toBe(false)
  })

  it('uses the displayed-frame counter when video callbacks are unavailable', () => {
    let frames = 0
    const video = {
      currentTime: 0,
      getVideoPlaybackQuality() { return { totalVideoFrames: frames } },
    } as unknown as HTMLVideoElement
    const gate = new CameraFrameGate()
    gate.start(video, 1)
    video.currentTime = 0.02
    expect(gate.consume(1.02)).toBe(false)
    frames = 1
    expect(gate.consume(1.03)).toBe(true)
    expect(gate.consume(1.04)).toBe(false)
  })

  it('does not mistake a dropped frame for a displayed frame', () => {
    let total = 0
    let dropped = 0
    const video = {
      currentTime: 0,
      getVideoPlaybackQuality() { return { totalVideoFrames: total, droppedVideoFrames: dropped } },
    } as unknown as HTMLVideoElement
    const gate = new CameraFrameGate()
    gate.start(video, 1)
    total = 1
    expect(gate.consume(1.03)).toBe(true)
    total = 2
    dropped = 1
    expect(gate.consume(1.06)).toBe(false)
  })

  it('recovers if a browser stops issuing callbacks for an invisible video', () => {
    const camera = cameraWithCallbacks()
    const gate = new CameraFrameGate()
    gate.start(camera.video, 1)
    // The image count moves, but the callback never arrives.
    camera.video.currentTime = 1 / 30
    expect(gate.consume(1.05)).toBe(false)
    // Advance only the counter without invoking the callback.
    const fallback = camera.video as HTMLVideoElement & { getVideoPlaybackQuality(): { totalVideoFrames: number } }
    fallback.getVideoPlaybackQuality = () => ({ totalVideoFrames: 1 })
    expect(gate.consume(1.11)).toBe(true)
    expect(gate.consume(1.12)).toBe(false)
  })
})
