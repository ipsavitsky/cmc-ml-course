class BankCard:
    def __init__(self, sum_on_card):
        self.total_sum = sum_on_card
        pass

    def __repr__(self):
        return "To learn the balance you should put the money on the card, spent some money or get the bank data. The last procedure is not free and costs 1 dollar."

    @property
    def balance(self):
        if self.total_sum - 1 < 0:
            print("Not enough money to learn balance.")
            raise ValueError
        self.total_sum -= 1
        return self.total_sum

    def __call__(self, sum_to_take):
        if sum_to_take > self.total_sum:
            print("Not enough money to spent sum_spent dollars.")
            raise ValueError
        self.total_sum -= sum_to_take
        print(f"You spent {sum_to_take} dollars. {self.total_sum} dolars are left")
        pass

    def put(self, sum_put):
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars. {self.total_sum} dollars are left")
