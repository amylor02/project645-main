import argparse
import copy

import train_vq_vae

GENERATOR = 1
IK = 2


SMOKE_TEST_OVERRIDES = {
    "epochs": 1,
    "batch_size": 1,
    "window_step": 128,
    "random_yaw_augmentation": False,
}


def resolve_train_mode(train_mode):
    if train_mode == "generator":
        return GENERATOR
    if train_mode == "ik":
        return IK
    if train_mode == "all":
        return GENERATOR | IK
    raise ValueError("Unsupported train mode {}".format(train_mode))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("name", type=str)
    parser.add_argument(
        "--train_mode",
        type=str,
        default="generator",
        choices=["generator", "ik", "all"],
    )
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--window_step", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--stride_encoder_conv", type=int, default=None)
    parser.add_argument("--bvh_scale_factor", type=float, default=None)
    parser.add_argument("--autoencoder_module", type=str, default=None)
    parser.add_argument(
        "--training_stage", type=str, default=None, choices=["ae", "vq_vae", "rnn"]
    )
    parser.add_argument("--vae_latent_dim", type=int, default=None)
    parser.add_argument("--sparse_joints", type=int, nargs="+", default=None)
    parser.add_argument("--feet_idxs", type=int, nargs="+", default=None)
    parser.add_argument("--shoulder_idxs", type=int, nargs=2, default=None)
    parser.add_argument("--head_idx", type=int, default=None)
    parser.add_argument("--skeleton_height", type=float, default=None)
    parser.add_argument("--head_height", type=float, default=None)
    parser.add_argument("--feet_contact_threshold", type=float, default=None)
    parser.add_argument("--disable_vae", action="store_true")
    parser.add_argument("--disable_random_yaw_augmentation", action="store_true")
    args = parser.parse_args()

    train_vq_vae.param = copy.deepcopy(train_vq_vae.param)
    train_vq_vae.param2 = train_vq_vae.param

    if args.smoke_test:
        train_vq_vae.param.update(SMOKE_TEST_OVERRIDES)

    if args.epochs is not None:
        train_vq_vae.param["epochs"] = args.epochs
    if args.batch_size is not None:
        train_vq_vae.param["batch_size"] = args.batch_size
    if args.window_size is not None:
        train_vq_vae.param["window_size"] = args.window_size
    if args.window_step is not None:
        train_vq_vae.param["window_step"] = args.window_step
    if args.learning_rate is not None:
        train_vq_vae.param["learning_rate"] = args.learning_rate
    if args.stride_encoder_conv is not None:
        train_vq_vae.param["stride_encoder_conv"] = args.stride_encoder_conv
    if args.bvh_scale_factor is not None:
        train_vq_vae.param["bvh_scale_factor"] = args.bvh_scale_factor
    if args.autoencoder_module is not None:
        train_vq_vae.param["autoencoder_module"] = args.autoencoder_module
    if args.training_stage is not None:
        train_vq_vae.param["training_stage"] = args.training_stage
    if args.vae_latent_dim is not None:
        train_vq_vae.param["vae_latent_dim"] = args.vae_latent_dim
        train_vq_vae.param["vae_hidden_dim"] = args.vae_latent_dim
    if args.sparse_joints is not None:
        train_vq_vae.param["sparse_joints"] = list(args.sparse_joints)
    if args.feet_idxs is not None:
        train_vq_vae.param["feet_idxs"] = list(args.feet_idxs)
    if args.shoulder_idxs is not None:
        train_vq_vae.param["shoulder_idxs"] = list(args.shoulder_idxs)
    if args.head_idx is not None:
        train_vq_vae.param["head_idx"] = args.head_idx
    if args.skeleton_height is not None:
        train_vq_vae.param["skeleton_height"] = args.skeleton_height
    if args.head_height is not None:
        train_vq_vae.param["head_height"] = args.head_height
    if args.feet_contact_threshold is not None:
        train_vq_vae.param["feet_contact_threshold"] = args.feet_contact_threshold
    if args.disable_vae:
        train_vq_vae.param["use_vae"] = False
    if args.disable_random_yaw_augmentation:
        train_vq_vae.param["random_yaw_augmentation"] = False

    print("Running my_autoencoder with param summary:")
    for key in [
        "epochs",
        "batch_size",
        "window_size",
        "window_step",
        "learning_rate",
        "stride_encoder_conv",
        "bvh_scale_factor",
        "training_stage",
        "autoencoder_module",
        "use_vae",
        "vae_latent_dim",
        "random_yaw_augmentation",
        "sparse_joints",
        "feet_idxs",
        "shoulder_idxs",
        "head_idx",
        "skeleton_height",
        "feet_contact_threshold",
    ]:
        print("{}: {}".format(key, train_vq_vae.param.get(key)))

    train_vq_vae.main(
        argparse.Namespace(
            data_path=args.data_path,
            name=args.name,
            train_mode=resolve_train_mode(args.train_mode),
            load=args.load,
            bvh_scale_factor=args.bvh_scale_factor,
        )
    )


if __name__ == "__main__":
    main()
