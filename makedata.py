from pydub import AudioSegment
import os


def calculate_total_duration(input_folder):
    total_duration = 0
    print("Input files")
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            print(f"{filename}")
            input_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_mp3(input_path)
            total_duration += len(audio)
    return total_duration


def concat_and_split_mp3(input_folder, output_folder, num_segments):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    concatenated_audio = AudioSegment.silent(duration=0)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            input_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_mp3(input_path)
            concatenated_audio += audio

    segment_length = len(concatenated_audio) // num_segments

    for i in range(num_segments):
        start_time = i * segment_length
        end_time = (i + 1) * segment_length
        segment = concatenated_audio[start_time:end_time]

        output_filename = f"segment_{i + 1}.mp3"
        output_path = os.path.join(output_folder, output_filename)
        segment.export(output_path, format="mp3")


if __name__ == "__main__":
    input_folder = "./datavoice"
    output_folder = "./data"
    num_segments = 128

    total_duration = calculate_total_duration(input_folder)
    segment_duration = total_duration // num_segments

    concat_and_split_mp3(input_folder, output_folder, num_segments)
