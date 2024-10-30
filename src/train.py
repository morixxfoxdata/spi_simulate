from utils.data_process import npz_data_mnist


def main():
    # gen1 = generate_mask_pattern(
    #     time_length=10, num_x_pixel_true=256, num_y_pixel_true=256
    # )
    # print(gen1.shape)
    npz_data_mnist(9)


if __name__ == "__main__":
    main()
