from lob.data import align_events, load_messages, load_orderbook, validate_book

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

events = align_events(messages, book)

assert events.shape == (400391, 46)
assert events.index.name == "event_id"

print(events[["order_id", "bid_price_1", "bid_size_1"]].head())

checks = validate_book(book, n_levels=10)

print("\nOrder book validation:")
print(checks.sum())
