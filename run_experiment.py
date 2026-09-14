from lob.data import load_messages, load_orderbook

messages = load_messages("data/raw/AAPL_2012-06-21_34200000_57600000_message_10.csv")

print(messages.head())
print(messages.shape)
print(messages.dtypes)
print(messages["event_type"].value_counts())

book = load_orderbook("data/raw/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv")

print(book.shape)
print(book.iloc[:5, :8])

assert len(messages) == len(book)
assert book.columns.is_unique
