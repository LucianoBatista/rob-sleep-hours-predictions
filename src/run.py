import yaml
from data_prep.prep import prep_data


def run():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_path = config["data"]["train_path"]
    prep_data(train_path)


if __name__ == "__main__":
    run()
