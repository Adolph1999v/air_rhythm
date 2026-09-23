import { describe, expect, it } from 'vitest'
import { FingertipMotionTracker, HandStabilizer } from '../src/hands'
import type { TrackedHand } from '../src/tracking'

function hand(label: string, x: number, y = 0.5): TrackedHand {
  return {
    label, confidence: 0.99,
    landmarks: Array.from({ length: 21 }, () => ({ x, y, z: 0 })),
  }
}

describe('smoothed hands and fresh-fingertip interaction', () => {
  it('preserves left/right identity when the model reverses its result order', () => {
    const stabilizer = new HandStabilizer()
    expect(stabilizer.update([hand('Left', 0.2), hand('Right', 0.8)], 1).map(item => item.identity)).toEqual(['Left', 'Right'])
    const active = stabilizer.update([hand('Right', 0.81), hand('Left', 0.21)], 1.03)
    expect(active.map(item => item.identity)).toEqual(['Left', 'Right'])
    expect(active[0].landmarks[8].x).toBeGreaterThan(0.2)
    expect(active[0].landmarks[8].x).toBeLessThan(0.21)
  })

  it('briefly holds a lost stick on screen without making a stale hand interactive', () => {
    const stabilizer = new HandStabilizer()
    const motions = new FingertipMotionTracker()
    const first = stabilizer.update([hand('Left', 0.2)], 1)
    expect(motions.update(first, 1, 1000, 600)).toHaveLength(5)
    const missing = stabilizer.update([], 1.03)
    expect(missing).toHaveLength(0)
    expect(motions.update(missing, 1.03, 1000, 600)).toHaveLength(0)
    expect(stabilizer.visibleAt(1.10)).toHaveLength(1)
    expect(stabilizer.visibleAt(1.15)).toHaveLength(0)
  })

  it('gives plain contact a Good hit without requiring a down or forward gesture', () => {
    const motions = new FingertipMotionTracker()
    const samples = motions.update([{ ...hand('Left', 0.4), identity: 'Left' }], 1, 1000, 600)
    expect(samples).toHaveLength(5)
    expect(samples[0].rating).toBe('GOOD')
    expect(samples[0].previous).toEqual(samples[0].current)
  })
})
