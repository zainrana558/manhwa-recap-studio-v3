import { test, expect } from 'bun:test'
import { clean_ocr_text_with_sfx } from '../mini-services/pipeline-service/lib'

test('clean_ocr_text_with_sfx removes garbage and normalizes sfx', () => {
  expect(clean_ocr_text_with_sfx('BOOM! xz What is happening?')).toBe('*BOOM!* What is happening?')
  expect(clean_ocr_text_with_sfx('SWOOSH ll1')).toBe('*SWOOSH!*')
  expect(clean_ocr_text_with_sfx('random 1-2 char fragment zq kj')).toBe('random char fragment')
  expect(clean_ocr_text_with_sfx('I am going to the dungeon.')).toBe('I am going to the dungeon.')
})
