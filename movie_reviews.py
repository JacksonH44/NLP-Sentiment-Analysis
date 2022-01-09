# Class to test any sentiment
import sentiment as sent

# TODO: Next steps are to use twitter API to livestream tweets and classify as positive or negative
if __name__ == '__main__':
    print(sent.classify_out("Wow this project is awesome!"))
