from utils.speckle_generate import generate_mask_pattern


def main():
    gen1 = generate_mask_pattern()
    print(gen1.shape)


if __name__ == "__main__":
    main()
