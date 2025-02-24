def twoSum(nums, target):
    for i in range(len(nums) - 1):
        if target - nums[i] in nums[i + 1:]:
            return [i, nums[i + 1:].index(target - nums[i])+1+i]
nums = [3,3]
target = 6
footballers_goals = {'Eusebio': 120, 'Cruyff': 104, 'Pele': 150, 'Ronaldo': 132, 'Messi': 125}
l = sorted(footballers_goals.items(), key=lambda x: x[1], reverse=False)
print(l)
# print(twoSum(nums, target))
