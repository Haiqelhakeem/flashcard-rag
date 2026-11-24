from flashcard_rag import generate_flashcard_data
# Run a test query
topic = "fotosintesis"
print(f"Testing topic: {topic}")
result = generate_flashcard_data(topic)

if result:
    print(f"\n🎉 Success! Got {len(result['flashcards'])} flashcards.")
else:
    print("\n No flashcards generated (Threshold might be too strict).")