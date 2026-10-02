import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


print("Loading Qwen3-TTS...")

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

print("Model loaded.")

wavs, sample_rate = model.generate_voice_clone(
    text=(
        "это тест аудио книги раз раз раз как слышно ? бла бла бла бла ."
    ),
    language="Russian",

    # Пока здесь потребуется reference voice.
    ref_audio="reference.wav",
    ref_text="Здесь было трудно кого-нибудь удивить. Близость с сенной, обилие известных заведений и по преимуществу цеховое и ремесленное население, скученное в этих серединных петербургских улицах и переулках.",
)

sf.write("test.wav", wavs[0], sample_rate)

print(f"Done: test.wav ({sample_rate} Hz)")