import whisperx
import gc 
import runpod


# function to do all steps in one go

def transcribe(job):
    input = job["input"]
    audio = input["audio"]
    device = "cuda" 
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16"
    model = whisperx.load_model("medium", device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size)
    return result

runpod.serverless.start({"handler": transcribe})