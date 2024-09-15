from BDIAgent import SimpleAgent, SimpleBelief, SimpleDesire, SimpleIntention

class BookAgent(SimpleAgent):
    def __init__(self, acoes):
        super().__init__(acoes=acoes, config_file_name='book_agent_config.json')

    def create_belief(self, config):
        if config['name'] == 'BookAvailable':
            return SimpleBelief(config['name'], config['description'], config['value'])
        elif config['name'] == 'BookInCollection':
            return SimpleBelief(config['name'], config['description'], config['value'])
        else:
            raise ValueError(f"Belief {config['name']} not recognized")

    def create_desire(self, config):
        if config['name'] == 'FindBook':
            return SimpleDesire(config['name'], config['description'])
        else:
            raise ValueError(f"Desire {config['name']} not recognized")

    def create_intention(self, name, action, conditions, goals):
        if name == 'FindBook':
            return SimpleIntention(name, action, conditions, goals)
        else:
            raise ValueError(f"Intention {name} not recognized")

    def add_belief(self, belief):
        self.beliefs.append(belief)

    def add_desire(self, desire):
        self.desires.append(desire)

    def add_intention(self, intention):
        self.intentions.append(intention)

    def execute_intentions(self):
        for intention in self.intentions:
            intention.perform()

    def search_book(self, book_title):
        # Simulate searching for a book
        
        print(f"Is book '{book_title}' available?")
        book_available = SimpleBelief('BookAvailable', f"Is book '{book_title}' available?", True)
        
        print(f"Is book '{book_title}' in collection?")
        book_in_collection = SimpleBelief('BookInCollection', f"Is book '{book_title}' in collection?", True)

        self.add_belief(book_available)
        self.add_belief(book_in_collection)

        print(f"Find book '{book_title}'")
        find_book = SimpleDesire('FindBook', f"Find book '{book_title}'", False)
        self.add_desire(find_book)

        intention = SimpleIntention('FindBook', self.acoes['acao1'], [book_available], [find_book])
        self.add_intention(intention)

        self.execute_intentions()


def acao1():
    print("Searching for book...")
    print("...")
    print("Book found!")

acoes = {
    'acao1': acao1
}

agent = BookAgent(acoes)
agent.search_book('Harry Potter')