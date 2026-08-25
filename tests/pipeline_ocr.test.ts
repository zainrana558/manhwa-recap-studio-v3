import { test, expect } from 'bun:test'
import { generateImageNarrationsOCR } from '../mini-services/pipeline-service/lib'

test('generateImageNarrationsOCR sequential processing and summary log', async () => {
  // Mock fetch to simulate PaddleOCR service endpoint
  const originalFetch = globalThis.fetch
  let requestCount = 0
  const requestedBodies: any[] = []

  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    if (url.toString().includes('/ocr/batch')) {
      requestCount++
      const body = JSON.parse(init?.body as string)
      requestedBodies.push(body)
      return new Response(
        JSON.stringify({
          results: [
            {
              index: 0,
              text: 'Solo Leveling Chapter 1 Panel Text',
              confidence: 0.95,
              regions: 1,
              status: 'SUCCESS',
            },
          ],
          model: 'PP-OCRv5',
          processing_time_ms: 120,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }
    return originalFetch(url, init)
  }) as typeof fetch

  try {
    const mockPanels = [
      '/tmp/solo_leveling_ch1_panel_1.png',
      '/tmp/solo_leveling_ch1_panel_2.png',
      '/tmp/solo_leveling_ch1_panel_3.png',
    ]

    const outcome = await generateImageNarrationsOCR(mockPanels)

    expect(requestCount).toBe(3)
    expect(requestedBodies.length).toBe(3)
    expect(requestedBodies[0].images.length).toBe(1)
    expect(outcome.results.length).toBe(3)
    expect(outcome.results[0].text).toBe('Solo Leveling Chapter 1 Panel Text')
  } finally {
    globalThis.fetch = originalFetch
  }
})
