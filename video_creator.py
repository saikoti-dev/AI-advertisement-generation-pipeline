import os
from pathlib import Path

from moviepy.editor import (
    AudioFileClip,
    VideoFileClip,
    concatenate_videoclips,
)


def create_video_with_audio(
    video_files: list[str | Path],
    audio_file: str | Path,
    output_filename: str = "final_ad.mp4",
    transition_duration: float = 1.0,
) -> str:
    """
    Stitches a list of video clips together with crossfade transitions,
    then lays the supplied audio track underneath.

    If the video is longer than the audio, it is trimmed to match.
    If the audio is longer than the video, the video holds at the end.

    Args:
        video_files:         Ordered list of .mp4 paths to combine.
        audio_file:          Path to the audio (.wav or .mp3).
        output_filename:     Destination .mp4 path.
        transition_duration: Crossfade length in seconds between clips.

    Returns:
        Absolute path to the finished video.
    """
    print(f"Stitching {len(video_files)} clip(s) with audio: {audio_file}…")

    audio = AudioFileClip(str(audio_file))

    clips = []
    for vf in video_files:
        vf = str(vf)
        if os.path.exists(vf):
            clips.append(VideoFileClip(vf))
        else:
            print(f"  ⚠ Video not found, skipping: {vf}")

    if not clips:
        raise FileNotFoundError("No valid video clips found — cannot create final video.")

    # Apply crossfade to every clip after the first
    faded = [clips[0]]
    for clip in clips[1:]:
        faded.append(clip.crossfadein(transition_duration))

    final_video = concatenate_videoclips(
        faded, method="compose", padding=-transition_duration
    )

    # Trim video to audio length if needed
    if final_video.duration > audio.duration:
        final_video = final_video.subclip(0, audio.duration)

    final_video = final_video.set_audio(audio)

    print(f"Writing final video → {output_filename}")
    final_video.write_videofile(output_filename, codec="libx264", audio_codec="aac")

    for clip in clips:
        clip.close()
    audio.close()
    final_video.close()

    print(f"✓ Final video saved: {output_filename}  ({final_video.duration:.1f}s)")
    return os.path.abspath(output_filename)
