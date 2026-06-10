class ImprovementChecker:
    """
    class used to determine if there's an improvement in network performance by checking loss and accuracy values
    """
    def __init__(self, db, lfi):
        self.db = db
        self.lfi = lfi

    def checker(self, last_score):
        """
        method that analyses trends in metrics and determines whether there was an improvement
        :param last_score: current score value used to compare with values from past iterations
        :return: boolean indicating if there was an improvement
        """
        # obtain the values from db
        score = self.db.get()
        print("[DEBUG] Improvement Checker - Score history and last score:")
        print("Score history: ", score)
        print("Last score: ", last_score)
        # if the length of the score history is 0, nothing has been saved yet in the db
        if len(score) == 0:
            return None

        # set to true that there was an improvement in score and accuracy
        score_check = True

        # if there's a degradation compared to the last training
        if last_score > score[len(score) - 1]:
            score_check = False
        print("[Debug] Improvement Checker - Score check result: ", score_check)
        return score_check