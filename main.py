from load_data import show_dataset_summary
from train_model import train_models


def main():
    print("Predictive Maintenance of Industrial Machines")
    print("=" * 48)
    show_dataset_summary()
    train_models()


if __name__ == "__main__":
    main()
