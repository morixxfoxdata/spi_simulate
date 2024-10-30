from utils.speckle_generate import generate_mask_pattern


def main():
    gen1 = generate_mask_pattern(time_length=10, num_x_pixel=200, num_y_pixel=200)
    print(gen1.shape)


if __name__ == "__main__":
    main()
