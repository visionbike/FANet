from argparse import ArgumentParser
from data.preprocessing import PreprocessorAsr


if __name__ == "__main__":
    # load arguments
    parser = ArgumentParser(description="Processing ASRGesture dataset")
    parser.add_argument("--root", type=str, help="The raw ASRGesture dataset path")
    parser.add_argument("--save", type=str, help="The save path")
    parser.add_argument("--step_size", type=int, default=45, help="step size")
    parser.add_argument("--window_size", type=int, default=150, help="window size")
    parser.add_argument("--dc", action="store_true", default=False, help="Using zero-mean shifting")
    parser.add_argument("--rectify", action="store_true", default=False, help="Using signal rectifying")
    parser.add_argument("--lowpass", action="store_true", default=False, help="Using butterworth lowpass filter for smooth")
    parser.add_argument("--highpass", action="store_true", default=False, help="Using butterworth highpass filter for noise removal")
    parser.add_argument("--minmax", action="store_true", default=False, help="Using min-max normalization")
    parser.add_argument("--first", action="store_true", default=False, help="Using first appearance")
    parser.add_argument("--relax", action="store_true", default=False, help="Using 'Relax' label")
    parser.add_argument("--quantize", action="store_true", default=False, help="Applying quantization")
    parser.add_argument("--multiproc", action="store_true", default=False, help="Using multi-processing")
    parser.add_argument("--split", type=str, default="trainvaltest", help="Split data")
    parser.add_argument("--valtest_reps", type=str, default="", help="Sessions for valtest set")
    parser.add_argument("--exercise", type=int, default=-1, help="Exercise used to load")
    args = parser.parse_args()

    # create paths
    folder = f"s{args.step_size}_"
    folder += f"w{args.window_size}_"
    if args.dc_filter:
        folder += "dc_"
    if args.rectify:
        folder += "rectify_"
    if args.lowpass_filter:
        folder += "lowpass_"
    if args.highpass_filter:
        folder += "highpass_"
    if args.minmax:
        folder += "minmax_"
    if args.first:
        folder += "first_"
    else:
        folder += "major_"
    if args.quantize:
        folder += "quantize_"
    if args.relax:
        folder += "relax"
    else:
        folder += "no_relax"

    # initiate preprocessing
    processor = PreprocessorAsr(data_root=args.root,
                                step_size=args.step_size,
                                window_size=args.window_size,
                                dc=args.dc,
                                rectif=args.rectify,
                                lowpass=args.lowpass,
                                highpass=args.highpass,
                                minmax=args.minmax,
                                first_appearance=args.first,
                                quantize=args.quantize,
                                use_relax_label=args.relax,
                                exercise=args.exercise)
    # process data
    processor.load_data()
    processor.process_data(args.multiproc)
    # split data
    valtest_reps = [int(i) for i in args.valtest_reps.split(",")] if args.valtest_reps else None
    processor.split_data(mode=args.split, save_path=f"{args.save}/{folder}", valtest_reps=valtest_reps)

