def pie_autopct(data):
    """
    Function for automatic labeling of pie charts bu num
    :param percent:
    :return:
    """
    def autopct(percent):
        total = sum(data)
        value = int(round(percent * total / 100.0))
        return f'{round(percent, 2)}% ({value})'
    return autopct
