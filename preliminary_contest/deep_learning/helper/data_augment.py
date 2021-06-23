import random


class DataAugment(object):
    """
    训练集数据增强，需要增强时就在处理DataFrame的时候加上增强数据
    """

    @staticmethod
    def segment_shuffle(text):
        """
        分段乱序
        """
        shuffle_box = []

        input_text = list(map(int, text))

        length = len(input_text)

        shuffle_box.append(input_text[:int(length / 3)])
        shuffle_box.append(input_text[int(length / 3):int(length * 2 / 3)])
        shuffle_box.append(input_text[int(length * 2 / 3):])

        random.shuffle(shuffle_box)

        sentence = []

        for segment in shuffle_box:
            for word in segment:
                sentence.append(int(word))

        return sentence

    @staticmethod
    def random_swap(text):
        """
        随机交换
        """

        def swap_word(text_copy):
            text_len = len(text_copy)

            random_idx_1 = random.randint(0, text_len - 1)
            random_idx_2 = random_idx_1

            counter = 0

            while random_idx_2 == random_idx_1:
                random_idx_2 = random.randint(0, text_len - 1)
                counter += 1

                if counter > 3:
                    return text_copy

            text_copy[random_idx_1], text_copy[random_idx_2] = text_copy[random_idx_2], text_copy[random_idx_1]

            return text_copy

        text = [int(word) for word in text]

        new_text = text.copy()

        new_text_len = len(new_text)

        for _ in range(int(new_text_len / 2)):
            new_text = swap_word(new_text)

        return new_text

    @staticmethod
    def random_deletion(text):
        """
        随机删除
        """
        text = [int(word) for word in text]

        if len(text) == 1:
            return text

        new_text = []

        for word in text:
            r = random.uniform(0, 1)

            if r > 0.3:
                new_text.append(word)

        return new_text
