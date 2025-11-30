import os
import imageio.v2 as imageio

def main():
    frames_dir = "sidereal_axis_frames"
    output = "sidereal_axis_movie.mp4"

    # collect all png frames
    frames = sorted(
        [os.path.join(frames_dir, f)
         for f in os.listdir(frames_dir)
         if f.endswith(".png")]
    )

    if not frames:
        print("no frames found — run sidereal_axis_map.py first")
        return

    print(f"found {len(frames)} frames, building movie...")

    writer = imageio.get_writer(output, fps=4)
    for f in frames:
        writer.append_data(imageio.imread(f))
    writer.close()

    print(f"done. saved → {output}")

if __name__ == "__main__":
    main()
