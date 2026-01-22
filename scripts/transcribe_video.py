from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_id
from videorag.pipeline.asr_transcription import transcribe_video_to_derived


def main() -> None:
    video_id = pick_video_id("Select video to transcribe")
    paths = video_paths(video_id)

    raw_video_path = paths.raw_video_path
    if not raw_video_path.exists():
        print(f"Raw video not found: {raw_video_path}")
        print("Add the video again or check your raw folder structure.")
        return

    print(f"Transcribing '{video_id}'...")
    transcribe_video_to_derived(
        video_id=video_id,
        input_path=raw_video_path,
    )

    print(f"Transcription completed for '{video_id}'")
    print(f"Output in: {paths.derived_dir}")


if __name__ == "__main__":
    main()

