import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision'
import modelUrl from '../../models/hand_landmarker.task?url'

export interface TrackedHand {
  label: string
  landmarks: NormalizedLandmark[]
}

export async function createHandTracker(): Promise<HandLandmarker> {
  try {
    const fileset = await FilesetResolver.forVisionTasks(`${import.meta.env.BASE_URL}wasm`)
    return await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' },
      runningMode: 'VIDEO',
      numHands: 2,
    })
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    throw new Error(`The hand model could not load. Check the local web assets and try again. ${detail}`, { cause: error })
  }
}

export function trackedHandsFrom(result: HandLandmarkerResult): TrackedHand[] {
  return result.landmarks.slice(0, 2).flatMap((landmarks, index) => {
    if (landmarks.length < 21) return []
    return [{
      label: result.handedness[index]?.[0]?.categoryName ?? `Hand ${index + 1}`,
      landmarks,
    }]
  })
}
