import { describe, expect, it } from 'vitest'
import { RhythmGame, challengeNodeY, nodeRadius } from '../src/game'
import { motionTouchesCircle } from '../src/hands'
import { MELODY_STEPS, noteName } from '../src/music'
import { RhythmRound, RoundScore, buildMelodyChart, gradeTiming } from '../src/rhythm'

describe('desktop song and scoring parity', () => {
  it('preserves the 35-note melody, names, and stretched note spacing', () => {
    const chart = buildMelodyChart()
    expect(chart).toHaveLength(35)
    expect(chart.map(note => note.pitch)).toEqual(MELODY_STEPS.map(([pitch]) => pitch))
    expect(chart[0].targetOffset).toBe(2)
    expect(chart[1].targetOffset).toBeCloseTo(2 + 0.24 * 1.75)
    expect(chart[9].targetOffset - chart[8].targetOffset).toBeCloseTo(0.72 * 1.75)
    expect(noteName(76)).toBe('E5')
    expect(noteName(75)).toBe('D#5')
  })

  it('keeps the same inclusive Perfect, Great, and Good windows', () => {
    expect(gradeTiming(0.10)).toBe('PERFECT')
    expect(gradeTiming(-0.10)).toBe('PERFECT')
    expect(gradeTiming(0.1001)).toBe('GREAT')
    expect(gradeTiming(0.22)).toBe('GREAT')
    expect(gradeTiming(0.2201)).toBe('GOOD')
    expect(gradeTiming(-0.42)).toBe('GOOD')
    expect(gradeTiming(0.4201)).toBeNull()
  })

  it('matches combo, movement, miss, completion, and timing scores', () => {
    const score = new RoundScore()
    expect(score.recordHit('PERFECT', 'GOOD')).toBe(1500)
    expect(score.recordHit('GREAT', 'PERFECT')).toBe(1350)
    score.recordMiss()
    expect(score.recordHit('GOOD', 'GREAT')).toBe(1140)
    expect(score.score).toBe(3990)
    expect(score.maxCombo).toBe(2)
    expect(score.completionAccuracy).toBe(75)
    expect(score.timingAccuracy).toBe(56.25)
    expect(score.rank).toBe('A')
  })

  it('gives every touched circle a base hit, even outside the timing window', () => {
    const score = new RoundScore()
    for (const _ of MELODY_STEPS) score.recordHit('HIT')
    expect(score.score).toBe(40_900)
    expect(score.totalHits).toBe(35)
    expect(score.completionAccuracy).toBe(100)
    expect(score.rank).toBe('S')
  })

  it('marks skipped earlier notes and out-of-order touches as basic hits', () => {
    const chart = buildMelodyChart()
    const round = new RhythmRound(100, 3, 2, chart)
    expect(round.judgeHit(chart[2], round.targetTime(chart[2]), 'GOOD')).toBe('HIT')
    expect(round.judgeHit(chart[0], round.targetTime(chart[0]), 'GOOD')).toBe('HIT')
    expect(round.judgeHit(chart[1], round.targetTime(chart[1]), 'GOOD')).toBe('HIT')
    expect(round.judgeHit(chart[3], round.targetTime(chart[3]), 'GOOD')).toBe('PERFECT')
    expect(round.score.basicHits).toBe(3)
    expect(round.score.perfect).toBe(1)
  })
})

describe('falling notes and fingertip contact', () => {
  const width = 1280
  const height = 800

  it('spawns no song circle before the countdown reaches zero', () => {
    const game = new RhythmGame(() => 0.5)
    game.startChallenge(100)
    game.update(102.999, width, height)
    expect(game.nodes).toHaveLength(0)
    game.update(103, width, height)
    expect(game.nodes).toHaveLength(1)
    expect(game.nodes[0].yRatio).toBeCloseTo(nodeRadius(width, height) / height)
  })

  it('keeps an identical fall speed before and after the 30% beat point', () => {
    const game = new RhythmGame(() => 0.5)
    game.startChallenge(100)
    game.update(103, width, height)
    const node = game.nodes[0]
    const round = game.round!
    expect(nodeRadius(width, height)).toBe(46) // Python int(800 * 0.0585)
    const beforeSpeed = challengeNodeY(node, 104, width, height, round) - challengeNodeY(node, 103, width, height, round)
    const afterSpeed = challengeNodeY(node, 106, width, height, round) - challengeNodeY(node, 105, width, height, round)
    expect(beforeSpeed).toBeCloseTo(afterSpeed)
    expect(challengeNodeY(node, round.targetTime(node.event!), width, height, round)).toBeCloseTo(0.30)
  })

  it('catches a note after it passes the middle of the stage', () => {
    const game = new RhythmGame(() => 0.5)
    game.startChallenge(100)
    game.update(103, width, height)
    const node = game.nodes[0]
    const radiusRatio = nodeRadius(width, height) / height
    const speedRatio = (0.30 - radiusRatio) / 2
    const hitAt = 103 + (0.65 - radiusRatio) / speedRatio
    const center = { x: node.xRatio * width, y: 0.65 * height }
    const hits = game.update(hitAt, width, height, [{ previous: center, current: center, rating: 'GOOD', speed: 0 }])
    expect(hits.length).toBeGreaterThanOrEqual(1)
    expect(game.round!.score.totalHits).toBeGreaterThanOrEqual(1)
    expect(game.nodes.some(item => item.id === node.id)).toBe(false)
  })

  it('checks a complete fingertip sweep, not just the two endpoints', () => {
    expect(motionTouchesCircle({ previous: { x: 0, y: 50 }, current: { x: 100, y: 50 }, rating: 'GOOD', speed: 1 },
      { x: 50, y: 50 }, 10)).toBe(true)
    expect(motionTouchesCircle({ previous: { x: 0, y: 80 }, current: { x: 100, y: 80 }, rating: 'GOOD', speed: 1 },
      { x: 50, y: 50 }, 10)).toBe(false)
  })

  it('misses only after the entire circle leaves the bottom', () => {
    const game = new RhythmGame(() => 0.5)
    game.startChallenge(100)
    game.update(103, width, height)
    const node = game.nodes[0]
    const radiusRatio = nodeRadius(width, height) / height
    const speedRatio = (0.30 - radiusRatio) / 2
    const nearExit = 103 + (1 + radiusRatio - 0.001 - radiusRatio) / speedRatio
    game.update(nearExit, width, height)
    expect(game.nodes.some(item => item.id === node.id)).toBe(true)
    expect(game.round!.score.misses).toBe(0)
    game.update(nearExit + 0.02, width, height)
    expect(game.nodes.some(item => item.id === node.id)).toBe(false)
    expect(game.round!.score.misses).toBeGreaterThan(0)
  })

  it('shifts song time when the help screen pauses the game', () => {
    const game = new RhythmGame(() => 0.5)
    game.startChallenge(100)
    const target = game.round!.targetTime(game.round!.chart[0])
    game.setPaused(true, 101)
    game.update(110, width, height)
    expect(game.nodes).toHaveLength(0)
    game.setPaused(false, 111)
    expect(game.round!.targetTime(game.round!.chart[0])).toBe(target + 10)
  })

  it('caps a slow free-play update at the desktop limit instead of fast-forwarding notes', () => {
    const game = new RhythmGame(() => 0.5)
    game.startFree(100)
    game.update(100.01, width, height)
    const node = game.nodes[0]
    const initialY = node.yRatio
    game.update(101, width, height)
    expect(node.yRatio - initialY).toBeCloseTo(0.28 * 0.1)
  })
})
