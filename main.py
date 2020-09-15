from scripts import train

if __name__ == '__main__':
    train("data/train", batch_size=16, shuffle=True)