import os

DATA_ROOT = 'data/raw/utkface_images'

age_counts   = {age: 0 for age in range(101)}
gender_counts = {'Male': 0, 'Female': 0}
race_counts  = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Others'}
race_totals  = {name: 0 for name in race_counts.values()}

total_images = 0
age_sum = 0
skipped_age = 0
skipped_gender = 0
skipped_race = 0

for dirpath, _, filenames in os.walk(DATA_ROOT):
    images = [f for f in filenames if f.lower().endswith('.jpg')]
    total_images += len(images)
    for img in images:
        parts = img.split('_')
        try:
            age = int(parts[0])
            if 0 <= age <= 100:
                age_counts[age] += 1
                age_sum += age
        except (ValueError, IndexError):
            skipped_age += 1
            pass
        try:
            gender = parts[1]
            if gender == '0':
                gender_counts['Male'] += 1
            elif gender == '1':
                gender_counts['Female'] += 1
        except (ValueError, IndexError):
            skipped_gender += 1
            pass
        try:
            race = int(parts[2])
            if 0 <= race <= 4:
                race_totals[race_counts[race]] += 1
        except (ValueError, IndexError):
            skipped_race += 1
            pass

print("--- Age distribution ---")
for age, count in age_counts.items():
    if count > 0:
        print(f"  Age {age}: {count} images")

print("\n--- Summary ---")
print(f"Total images found: {total_images}")

ages_with_data = [age for age, count in age_counts.items() if count > 0]
print(f"\nMin age: {min(ages_with_data)}")
print(f"Max age: {max(ages_with_data)}")

total_aged = sum(age_counts.values())
print(f"Mean age: {age_sum / total_aged:.1f}" if total_aged > 0 else "Mean age: N/A")


print(f"\n--- Skipped images due to parsing issues ---")
print(f"Skipped age: {skipped_age}")
print(f"Skipped gender: {skipped_gender}")
print(f"Skipped race: {skipped_race}")


print("\n--- Gender distribution ---")
total_gender = sum(gender_counts.values())
for gender, count in gender_counts.items():
    pct = (count / total_gender * 100) if total_gender > 0 else 0
    print(f"  {gender}: {count} images ({pct:.1f}%)")

print("\n--- Race distribution ---")
total_race = sum(race_totals.values())
for race, count in race_totals.items():
    pct = (count / total_race * 100) if total_race > 0 else 0
    print(f"  {race}: {count} images ({pct:.1f}%)")
