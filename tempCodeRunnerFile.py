def gaschnig_heuristic(state):
    state = list(state)
    goal = list(GOAL_STATE)
    visited = [False] * 16
    cost = 0

    for i in range(16):
        if visited[i] or state[i] == goal[i] or state[i] == 0:
            continue

        cycle_start = i
        while not visited[cycle_start] and state[cycle_start] != goal[cycle_start]:
            visited[cycle_start] = True
            correct_tile = goal[cycle_start]
            cycle_start = state.index(correct_tile)
        cost += 1

    return cost