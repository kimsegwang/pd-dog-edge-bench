# [노트북 Windows 전용] Windows 내장 TTS 로 합성 음성 명령 세트를 만든다 (16kHz mono 16-bit wav).
#   powershell -ExecutionPolicy Bypass -File data\tools\make_tts_commands.ps1
#
# 목적: 마이크/실제 녹음 없이도 '발화 종료 → 명령 판정' 지연을 끝까지 측정하기 위한 파이프라인 검증용.
# 한계: 등록 화자와 시험 화자가 같은 TTS 목소리 → 화자 인식 정확도는 의미 없음 (지연 측정용).
#       실제 정확도는 data/tools/record_commands.py 로 녹음한 세트로 측정할 것.
#
# 결과 (data/tools/build_command_stream.py 입력 규칙과 동일):
#   data/voice/tts_heami/enroll/*.wav                        등록 화자(target) 등록용 문장
#   data/voice/tts_heami/utts/<speaker>__<command>__<nn>.wav  시험 발화. speaker=target|other, command=chulbal…|none
# 명령 키·말은 configs/voice.toml [commands.<키>] 와 맞춰야 함 (아래 $cmds)
param([string]$OutDir = "data/voice/tts_heami")

Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

function Say([string]$voice, [int]$rate, [string]$text, [string]$path) {
    $synth.SelectVoice($voice)
    $synth.Rate = $rate
    $synth.SetOutputToWaveFile($path, $fmt)
    $synth.Speak($text)
    $synth.SetOutputToNull()
}

$ko = ($synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq "ko-KR" } | Select-Object -First 1).VoiceInfo.Name
$en = ($synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like "en-*" } | Select-Object -First 1).VoiceInfo.Name
if (-not $ko) { throw "한국어 TTS 음성이 없음 (설정 > 시간 및 언어 > 음성 에서 한국어 음성 추가)" }
Write-Host "target(ko)=$ko  other(en)=$en"

Remove-Item -Recurse -Force "$OutDir/utts" -ErrorAction SilentlyContinue   # 이전 명령어 세트 제거
New-Item -ItemType Directory -Force "$OutDir/enroll", "$OutDir/utts" | Out-Null

# 등록 문장 (명령어와 다른 문장)
$enroll = @("안녕하세요, 저는 오늘도 로봇과 함께 운동을 합니다.",
            "아침에 일어나서 물을 한 잔 마시고 산책을 나갔어요.",
            "손을 천천히 들어 올리고 다섯까지 세어 봅시다.",
            "오늘 날씨가 맑아서 기분이 좋습니다.",
            "약은 식사 후 삼십 분에 먹으면 됩니다.")
$i = 0
foreach ($t in $enroll) { $i++; Say $ko 0 $t ("{0}/enroll/target_{1:D2}.wav" -f $OutDir, $i) }

# 명령어 × 말하기 속도 5단계 (Rate -3..+3)
$cmds = [ordered]@{ chulbal = "출발"; geotja = "걷자"; meomchwo = "멈춰"; anja = "앉아"; ireona = "일어나"; dowajwo = "도와줘" }
$rates = @(-3, -1, 0, 1, 3)
foreach ($k in $cmds.Keys) {
    $n = 0
    foreach ($r in $rates) { $n++; Say $ko $r $cmds[$k] ("{0}/utts/target__{1}__{2:D2}.wav" -f $OutDir, $k, $n) }
}
# 등록 화자의 명령 아닌 말 (명령으로 오인하면 안 됨)
$n = 0
# 뒤 4개는 '도와줘'와 발음이 비슷한 말 → 거짓 위급 알림(명령 아닌 말 → 도와줘) 확인용
foreach ($t in @("안녕하세요", "물 좀 줄래", "오늘 뭐 먹지", "텔레비전 켜 줘", "더워", "도라지", "돈 좀 줘", "도와줄까")) { $n++; Say $ko 0 $t ("{0}/utts/target__none__{1:D2}.wav" -f $OutDir, $n) }
# 다른 화자(영어 TTS) — 화자 거부/명령 거부 확인용
if ($en) {
    $n = 0
    foreach ($t in @("Let's go", "Stop", "Sit down", "Stand up", "Help me")) { $n++; Say $en 0 $t ("{0}/utts/other__none__{1:D2}.wav" -f $OutDir, $n) }
}
$synth.Dispose()
Write-Host "done: $OutDir"
