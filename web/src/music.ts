import musicData from './music-data.json'

export type InstrumentKey = 'keys' | 'bell' | 'pluck' | 'marimba'
export interface Instrument {
  key: InstrumentKey
  label: string
  color: string
  freestyleNote: number
}

export const MELODY_TITLE = musicData.title
export const INSTRUMENTS = musicData.instruments as Instrument[]
export const MELODY_STEPS: ReadonlyArray<readonly [number, number]> = musicData.melody.map(
  ([pitch, duration]) => [pitch, duration] as const,
)

export function noteName(midiNote: number): string {
  if (!Number.isInteger(midiNote) || midiNote < 0 || midiNote > 127) {
    throw new RangeError('A MIDI note must be between 0 and 127.')
  }
  const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  return `${names[midiNote % 12]}${Math.floor(midiNote / 12) - 1}`
}
