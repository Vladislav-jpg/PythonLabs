def count_it(sequence):
  counts = {}
  for char in sequence:
    num = int(char)
    counts[num] = counts.get(num, 0) + 1
  pairs = []

  for num, count in counts.items():
    pairs.append([count, num])

  pairs.sort(reverse=True)
  result = {}
  for count, num in pairs[:3]:
    result[num] = count
  return result

text = "314159265358979323846"
print(count_it(text))
