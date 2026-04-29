from flask import Flask, jsonify, request, render_template
import json, math, os

app = Flask(__name__)
DATA_PATH = os.path.join(os.path.dirname(__file__), 'agent_data.json')
HISTORY_PATH = os.path.join(os.path.dirname(__file__), 'agent_history.json')

def load_history():
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except:
        return {}

def get_agent_history(agent_id):
    history = load_history()
    timesteps = [10, 20, 30, 40, 50]
    progression = []
    for t in timesteps:
        key = str(t)
        if key in history:
            agent = next((a for a in history[key] if a['id'] == agent_id), None)
            if agent:
                progression.append({
                    'timestep': t,
                    'delays': agent['delays'],
                    'trend': agent['trend'],
                    'blocked_by': agent['blocked_by'],
                    'distance': agent['distance'],
                    'priority': agent['priority']
                })
    return progression

def load_agents():
    with open(DATA_PATH) as f:
        return json.load(f)

def get_agent_by_id(aid):
    return next((a for a in load_agents() if a['id'] == aid), None)

# ── WoE Engine ────────────────────────────────────────────────────────────────

def p_intervene(agent, evidence_type):
    agents = load_agents()
    max_delay = max(a['delays'] for a in agents) or 1
    max_priority = max(a['priority'] for a in agents) or 1
    trend_scores = {'critical':0.95,'worsening':0.75,'moderate':0.55,'minor':0.35,'clear':0.05}

    if evidence_type == 'delays':
        return agent['delays'] / max_delay
    elif evidence_type == 'distance':
        closeness = 1 - (agent['distance'] / 60)
        return closeness * 0.9 if agent['delays'] > 5 else closeness * 0.3
    elif evidence_type == 'priority':
        return 1 - (agent['priority'] / max_priority)
    elif evidence_type == 'trend':
        return trend_scores.get(agent.get('trend','clear'), 0.5)
    elif evidence_type == 'blocked_by':
        return 0.85 if agent.get('blocked_by',-1) != -1 else 0.1
    return 0.5

def woe_score(p, prior=0.5):
    p = max(0.001, min(0.999, p))
    return math.log((p/(1-p)) * ((1-prior)/prior))

def compute_woe(agent):
    evidence = [
        {'name':'delay_count',  'label':f"{agent['delays']} accumulated delays",
         'woe': woe_score(p_intervene(agent,'delays'))},
        {'name':'proximity',    'label':f"only {agent['distance']} steps from goal",
         'woe': woe_score(p_intervene(agent,'distance'))},
        {'name':'priority',     'label':f"priority score {agent['priority']}",
         'woe': woe_score(p_intervene(agent,'priority'))},
        {'name':'trend',        'label':f"trend: {agent.get('trend','unknown')}",
         'woe': woe_score(p_intervene(agent,'trend'))},
        {'name':'blocked_by',   'label':f"blocker: {'Agent '+str(agent['blocked_by']) if agent.get('blocked_by',-1)!=-1 else 'none'}",
         'woe': woe_score(p_intervene(agent,'blocked_by'))},
    ]
    supporting = sorted([e for e in evidence if e['woe'] > 0], key=lambda x: -x['woe'])
    refuting   = sorted([e for e in evidence if e['woe'] <= 0], key=lambda x: x['woe'])
    return {
        'evidence': evidence,
        'supporting': supporting,
        'refuting': refuting,
        'supporting_marker': supporting[0] if supporting else None,
        'refuting_marker':   refuting[0]   if refuting   else None,
    }

# ── Explanation Engine ────────────────────────────────────────────────────────

def explain(agent, mode='auto'):
    agents = load_agents()
    max_delay = max(a['delays'] for a in agents) or 1
    blocked_by = agent.get('blocked_by', -1)
    trend = agent.get('trend', 'clear')
    blocker_str = f" by Agent {blocked_by}" if blocked_by != -1 else ""

    if mode == 'template':
        if agent['delays'] > 10:
            return (f"Agent {agent['id']} is heavily delayed ({agent['delays']} delays, trend: {trend}) "
                    f"and has very low priority ({agent['priority']}). "
                    f"It is repeatedly yielding to other agents{blocker_str}.")
        elif agent['delays'] > 3:
            return (f"Agent {agent['id']} has {agent['delays']} accumulated delays "
                    f"and is waiting for higher-priority agents to pass.")
        elif agent['delays'] > 0:
            return f"Agent {agent['id']} has {agent['delays']} delay(s) and is yielding to agents with higher priority."
        return f"Agent {agent['id']} has no delays and moves freely with priority {agent['priority']:,}."

    elif mode == 'contrastive':
        if agent['delays'] > 0:
            hyp = max_delay * 10000 + agent['distance']
            blocker_note = f" Currently being blocked{blocker_str}." if blocked_by != -1 else ""
            return (f"Agent {agent['id']} waited rather than moved because its {agent['delays']} "
                    f"accumulated delays reduced its priority to {agent['priority']}."
                    f"{blocker_note} "
                    f"If it had 0 delays, its priority would be {hyp:,} — placing it among the top movers.")
        return f"Agent {agent['id']} moved rather than waited — it has 0 delays and priority {agent['priority']:,}."

    else:  # causal / auto
        if agent['delays'] > 10:
            blocker_note = f" Currently, Agent {blocked_by} is occupying its next cell." if blocked_by != -1 else ""
            return (f"Agent {agent['id']} has been systematically blocked {agent['delays']} times "
                    f"(trend: {trend}).{blocker_note} "
                    f"It is only {agent['distance']} steps from its goal but cannot advance. "
                    f"This suggests a high-traffic corridor. "
                    f"Recommendation: reroute Agent {agent['id']} or reassign Agent {blocked_by}." 
                    if blocked_by != -1 else
                    f"Agent {agent['id']} has been systematically blocked {agent['delays']} times "
                    f"(trend: {trend}). It is only {agent['distance']} steps from its goal. "
                    f"Recommendation: reroute this agent.")
        elif agent['delays'] > 3:
            return (f"Agent {agent['id']} accumulated {agent['delays']} delays (trend: {trend}). "
                    f"It is {agent['distance']} steps from its goal. "
                    f"{'Blocked by Agent '+str(blocked_by)+'.' if blocked_by!=-1 else 'No specific blocker identified.'}")
        elif agent['delays'] > 0:
            return f"Agent {agent['id']} has {agent['delays']} delay(s). Likely temporary congestion."
        return f"Agent {agent['id']} is clear — 0 delays, priority {agent['priority']:,}, moving freely."

# ── Path Explanation ──────────────────────────────────────────────────────────

def explain_path(agent):
    pos = agent['position']
    goal = agent['goal']
    cols = 64
    curr_row, curr_col = pos // cols, pos % cols
    goal_row, goal_col = goal // cols, goal % cols
    manhattan = abs(goal_row - curr_row) + abs(goal_col - curr_col)
    actual = agent['distance']
    detour = actual - manhattan if actual > manhattan else 0

    agents = load_agents()
    pos_set = {a['position']: a['id'] for a in agents if a['id'] != agent['id']}

    # Check what's in the direct path
    obstacles = []
    r, c = curr_row, curr_col
    steps = 0
    while (r != goal_row or c != goal_col) and steps < 10:
        if goal_row > r: r += 1
        elif goal_row < r: r -= 1
        elif goal_col > c: c += 1
        else: c -= 1
        cell = r * cols + c
        if cell in pos_set:
            obstacles.append(pos_set[cell])
        steps += 1

    if detour > 0 and obstacles:
        return (f"Agent {agent['id']} is taking a path {detour} steps longer than optimal. "
                f"The direct route (Manhattan distance: {manhattan} steps) is occupied by "
                f"Agent(s) {obstacles[:3]}. "
                f"PIBT-D rerouted Agent {agent['id']} to avoid conflict, "
                f"adding {detour} extra steps to its journey.")
    elif detour > 0:
        return (f"Agent {agent['id']} is {actual} steps from goal but optimal is {manhattan}. "
                f"The {detour}-step detour was caused by corridor congestion — "
                f"higher-priority agents occupied the direct path at a previous timestep.")
    elif obstacles:
        return (f"Agent {agent['id']} is on the optimal path ({manhattan} steps) "
                f"but Agent(s) {obstacles[:3]} are directly ahead. "
                f"A delay is likely at the next timestep.")
    else:
        return (f"Agent {agent['id']} is on the optimal path — {actual} steps to goal "
                f"with no agents blocking the direct route. It should arrive without further delays.")

# ── Why Not Engine (Full Counterfactual) ─────────────────────────────────────

def simulate_path(start_pos, goal_pos, all_positions, cols=64, strategy='vertical_first'):
    """
    Simulate what would happen if agent took a different route.
    Returns: (path_cells, blocked_by, estimated_delays)
    """
    r, c = start_pos // cols, start_pos % cols
    goal_r, goal_c = goal_pos // cols, goal_pos % cols
    path = []
    delays = 0
    
    for _ in range(40):
        if r == goal_r and c == goal_c:
            break
            
        # Choose next move based on strategy
        if strategy == 'vertical_first':
            if r != goal_r:
                next_r = r + (1 if goal_r > r else -1)
                next_c = c
            else:
                next_r = r
                next_c = c + (1 if goal_c > c else -1)
        elif strategy == 'horizontal_first':
            if c != goal_c:
                next_r = r
                next_c = c + (1 if goal_c > c else -1)
            else:
                next_r = r + (1 if goal_r > r else -1)
                next_c = c
        elif strategy == 'diagonal':
            # Try to move both dimensions simultaneously
            next_r = r + (1 if goal_r > r else -1 if goal_r < r else 0)
            next_c = c + (1 if goal_c > c else -1 if goal_c < c else 0)
        
        next_pos = next_r * cols + next_c
        
        # Check if blocked
        if next_pos in all_positions:
            delays += 1
            path.append({'pos': r*cols+c, 'blocked': True, 'blocker': all_positions[next_pos]})
        else:
            r, c = next_r, next_c
            path.append({'pos': r*cols+c, 'blocked': False, 'blocker': None})
    
    return path, delays

def explain_why_not(agent):
    pos = agent['position']
    goal = agent['goal']
    cols = 64
    
    # Load history for richer analysis
    history = load_history()
    agents_now = load_agents()
    
    # Build position map (excluding this agent)
    pos_map = {a['position']: a['id'] for a in agents_now if a['id'] != agent['id']}
    
    # Simulate 3 alternative strategies
    strategies = {
        'vertical_first':   'Move vertically toward goal first, then horizontally',
        'horizontal_first': 'Move horizontally toward goal first, then vertically',
        'diagonal':         'Move diagonally (both dimensions simultaneously)'
    }
    
    results = {}
    for strategy, description in strategies.items():
        path, sim_delays = simulate_path(pos, goal, pos_map, cols, strategy)
        blocked_cells = [step for step in path if step['blocked']]
        results[strategy] = {
            'description': description,
            'simulated_delays': sim_delays,
            'blocked_count': len(blocked_cells),
            'blockers': list(set([s['blocker'] for s in blocked_cells if s['blocker']]))[:3],
            'path_length': len(path)
        }
    
    # Find best alternative
    best = min(results.items(), key=lambda x: x[1]['simulated_delays'])
    worst = max(results.items(), key=lambda x: x[1]['simulated_delays'])
    
    # Compare with actual situation
    actual_delays = agent['delays']
    best_delays = best[1]['simulated_delays']
    savings = actual_delays - best_delays
    
    # Build counterfactual explanation
    lines = []
    lines.append(f"Counterfactual analysis for Agent {agent['id']}:")
    lines.append(f"")
    
    for strategy, result in results.items():
        marker = "← BEST" if strategy == best[0] else ("← WORST" if strategy == worst[0] else "")
        lines.append(f"  {result['description']}:")
        if result['simulated_delays'] == 0:
            lines.append(f"    → 0 simulated delays — path appears clear {marker}")
        else:
            lines.append(f"    → {result['simulated_delays']} simulated delays, "
                        f"blocked by Agent(s) {result['blockers']} {marker}")
    
    lines.append(f"")
    
    # Historical counterfactual
    if '30' in history and savings > 0:
        agent_t30 = next((a for a in history['30'] if a['id'] == agent['id']), None)
        if agent_t30 and agent_t30['delays'] < 5:
            lines.append(f"Historical insight: At t=30, Agent {agent['id']} had only "
                        f"{agent_t30['delays']} delays and trend was '{agent_t30['trend']}'. "
                        f"If rerouted then using {best[1]['description'].lower()}, "
                        f"it could have avoided approximately {savings} of its current {actual_delays} delays.")
    elif savings > 0:
        lines.append(f"The {best[1]['description'].lower()} strategy "
                    f"would have resulted in approximately {savings} fewer delays. "
                    f"This suggests Agent {agent['id']} was routed into a high-traffic corridor "
                    f"that could have been avoided.")
    else:
        lines.append(f"All alternative routes show similar delay counts. "
                    f"Agent {agent['id']} appears to be in a genuinely congested area — "
                    f"corridor redesign rather than rerouting may be needed.")
    
    # Add WoE-style conclusion
    if best_delays == 0:
        lines.append(f"")
        lines.append(f"Recommendation: The {best[1]['description'].lower()} is currently clear. "
                    f"Rerouting Agent {agent['id']} now could resolve its critical state.")
    
    return '\n'.join(lines)

# ── Q&A Engine ────────────────────────────────────────────────────────────────

def answer_question(agent, question):
    q = question.lower()
    agents = load_agents()
    max_delay = max(a['delays'] for a in agents) or 1
    woe_data = compute_woe(agent)
    blocked_by = agent.get('blocked_by', -1)
    trend = agent.get('trend', 'clear')

    # WHY NOT questions
    if any(w in q for w in ['why not', 'alternative', 'different path', 'other route', 'instead', 'shortcut']):
        return {'answer': explain_why_not(agent), 'type': 'why_not', 'woe': woe_data}

    # PATH questions
    if any(w in q for w in ['path', 'route', 'shorter', 'why not', 'direction', 'detour']):
        return {'answer': explain_path(agent), 'type': 'path', 'woe': woe_data}

    # BLOCKED BY questions
    elif any(w in q for w in ['who', 'which agent', 'blocking', 'blocker', 'caused']):
        if blocked_by != -1:
            blocker = get_agent_by_id(blocked_by)
            blocker_info = f" Agent {blocked_by} has {blocker['delays']} delays and priority {blocker['priority']:,}." if blocker else ""
            answer = (f"Agent {agent['id']} is currently blocked by Agent {blocked_by}."
                     f"{blocker_info} "
                     f"Agent {blocked_by} has higher priority and occupies the next cell on Agent {agent['id']}'s path.")
        else:
            answer = (f"No specific agent is directly blocking Agent {agent['id']} right now. "
                     f"Its {agent['delays']} delays accumulated from corridor congestion over time — "
                     f"multiple agents passed through its path at different timesteps.")
        return {'answer': answer, 'type': 'blocker', 'woe': woe_data}

    # HISTORY questions
    elif any(w in q for w in ['history', 'over time', 'progression', 'when did', 'always', 'getting worse', 'timeline']):
        history = get_agent_history(agent['id'])
        if history:
            progression = ' → '.join([f"t={h['timestep']}:{h['delays']}delays({h['trend']})" for h in history])
            first = history[0]['delays']
            last  = history[-1]['delays']
            window = next((h['timestep'] for h in history if h['trend'] in ['worsening','critical']), None)
            answer = (f"Agent {agent['id']} delay progression: {progression}. "
                     f"Delays increased from {first} to {last} over 40 timesteps. "
                     f"{'Early intervention at t='+str(window)+' could have prevented critical state.' if window else 'Situation developed gradually.'}")
        else:
            answer = f"No historical data available for Agent {agent['id']}."
        return {'answer': answer, 'type': 'history', 'woe': woe_data}

    # TREND questions
    elif any(w in q for w in ['trend', 'getting worse', 'improving', 'pattern', 'history']):
        trend_msgs = {
            'critical':  f"Agent {agent['id']} is in CRITICAL state — {agent['delays']} delays. The situation is severe and worsening. Immediate intervention recommended.",
            'worsening': f"Agent {agent['id']} trend is WORSENING — {agent['delays']} delays and increasing. It is getting blocked more frequently over time.",
            'moderate':  f"Agent {agent['id']} shows MODERATE delays ({agent['delays']}). The situation is concerning but not yet critical.",
            'minor':     f"Agent {agent['id']} has MINOR delays ({agent['delays']}). Likely temporary — monitor but no immediate action needed.",
            'clear':     f"Agent {agent['id']} trend is CLEAR — 0 delays. It is performing optimally.",
        }
        return {'answer': trend_msgs.get(trend, f"Trend: {trend}"), 'type': 'trend', 'woe': woe_data}

    # WHY WAITING questions
    elif any(w in q for w in ['why', 'waiting', 'wait', 'stuck', 'blocked', 'stop']):
        if agent['delays'] > 10:
            return {'answer': explain(agent, 'causal'), 'type': 'causal', 'woe': woe_data}
        elif agent['delays'] > 0:
            return {'answer': explain(agent, 'contrastive'), 'type': 'contrastive', 'woe': woe_data}
        return {'answer': explain(agent, 'template'), 'type': 'template', 'woe': woe_data}

    # INTERVENE questions
    elif any(w in q for w in ['intervene', 'reroute', 'help', 'should i', 'action', 'fix']):
        sm = woe_data['supporting_marker']
        if agent['delays'] > 10:
            blocker_note = f" Specifically reroute Agent {blocked_by} or Agent {agent['id']}." if blocked_by != -1 else ""
            answer = (f"Strong evidence FOR intervention (WoE={sm['woe']:+.2f} for {sm['label']}). "
                     f"Trend is {trend}.{blocker_note} Recommend immediate action.")
        elif agent['delays'] > 3:
            answer = f"Moderate evidence for intervention. Trend: {trend}. Monitor closely — may resolve naturally."
        else:
            answer = f"No intervention needed. Agent {agent['id']} has {agent['delays']} delays and is performing normally."
        return {'answer': answer, 'type': 'intervention', 'woe': woe_data}

    # WHAT IF questions
    elif any(w in q for w in ['what if', 'if it had', '0 delay', 'no delay', 'hypothetical']):
        hyp = max_delay * 10000 + agent['distance']
        if agent['delays'] > 0:
            answer = (f"If Agent {agent['id']} had 0 delays, its priority would jump from "
                     f"{agent['priority']} to {hyp:,} (+{hyp-agent['priority']:,} points). "
                     f"It would rank among the top movers and pass freely through corridors "
                     f"where it currently yields.")
        else:
            answer = f"Agent {agent['id']} already has 0 delays and maximum priority {agent['priority']:,}."
        return {'answer': answer, 'type': 'counterfactual', 'woe': woe_data}

    # PRIORITY questions
    elif any(w in q for w in ['priority', 'score', 'rank']):
        all_sorted = sorted(agents, key=lambda x: x['priority'], reverse=True)
        rank = next((i+1 for i,a in enumerate(all_sorted) if a['id']==agent['id']), '?')
        answer = (f"Agent {agent['id']} has priority {agent['priority']:,}, "
                 f"ranked #{rank} of {len(agents)}. "
                 f"Formula: ({max_delay} - {agent['delays']}) × 10,000 + {agent['distance']} = {agent['priority']:,}.")
        return {'answer': answer, 'type': 'priority', 'woe': woe_data}

    # DISTANCE questions
    elif any(w in q for w in ['goal', 'far', 'distance', 'close', 'steps']):
        urgency = "urgent — it is close but stuck" if agent['distance'] < 15 and agent['delays'] > 5 else "normal"
        answer = f"Agent {agent['id']} is {agent['distance']} steps from its goal. Situation: {urgency}."
        return {'answer': answer, 'type': 'distance', 'woe': woe_data}

    else:
        return {'answer': explain(agent, 'auto'), 'type': 'general', 'woe': woe_data}

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/agents')
def get_agents():
    agents = load_agents()
    max_delay = max(a['delays'] for a in agents) or 1
    max_priority = max(a['priority'] for a in agents) or 1
    result = []
    for a in agents:
        woe_data = compute_woe(a)
        urgency = 'high' if a['delays'] > 10 else 'medium' if a['delays'] > 3 else 'low' if a['delays'] > 0 else 'none'
        result.append({**a, 'urgency': urgency,
            'supporting_marker': woe_data['supporting_marker'],
            'refuting_marker':   woe_data['refuting_marker'],
            'top_woe': max((e['woe'] for e in woe_data['evidence']), default=0)})
    result.sort(key=lambda x: x['delays'], reverse=True)
    return jsonify(result)

@app.route('/api/agent/<int:agent_id>')
def get_agent(agent_id):
    agent = get_agent_by_id(agent_id)
    if not agent: return jsonify({'error':'not found'}), 404
    woe_data = compute_woe(agent)
    history = get_agent_history(agent_id)
    return jsonify({**agent, 'woe': woe_data,
                    'explanation': explain(agent,'auto'),
                    'path_explanation': explain_path(agent),
                    'history': history})

@app.route('/api/explain/<int:agent_id>')
def get_explanation(agent_id):
    agent = get_agent_by_id(agent_id)
    if not agent: return jsonify({'error':'not found'}), 404
    return jsonify({
        'template':    explain(agent,'template'),
        'contrastive': explain(agent,'contrastive'),
        'causal':      explain(agent,'causal'),
        'path':        explain_path(agent),
        'woe':         compute_woe(agent)
    })

@app.route('/api/history/<int:agent_id>')
def get_history(agent_id):
    history = get_agent_history(agent_id)
    if not history:
        return jsonify({'error': 'No history found'}), 404
    
    # Compute trend analysis
    if len(history) >= 2:
        first = history[0]['delays']
        last  = history[-1]['delays']
        increase = last - first
        if increase > 15:
            analysis = f"Delays increased by {increase} over {len(history)} snapshots. Severe deterioration — should have been caught earlier."
        elif increase > 8:
            analysis = f"Delays increased by {increase} — significant worsening. Intervention window was between t={history[1]['timestep']} and t={history[2]['timestep']}."
        elif increase > 3:
            analysis = f"Delays increased by {increase} — moderate worsening. Situation developing but manageable."
        else:
            analysis = f"Minimal change ({increase} delay increase). Situation is stable."
    else:
        analysis = "Insufficient history for trend analysis."
    
    return jsonify({
        'agent_id': agent_id,
        'history': history,
        'analysis': analysis,
        'intervention_window': next(
            (h['timestep'] for h in history if h['trend'] in ['worsening','critical']),
            None
        )
    })

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json
    agent = get_agent_by_id(data.get('agent_id'))
    if not agent: return jsonify({'error':'not found'}), 404
    return jsonify(answer_question(agent, data.get('question','')))

@app.route('/api/stats')
def get_stats():
    agents = load_agents()
    trend_counts = {}
    for a in agents:
        t = a.get('trend','clear')
        trend_counts[t] = trend_counts.get(t,0) + 1
    return jsonify({
        'total_agents':   len(agents),
        'delayed_agents': len([a for a in agents if a['delays']>0]),
        'high_urgency':   len([a for a in agents if a['delays']>10]),
        'max_delay':      max(a['delays'] for a in agents),
        'avg_delay':      round(sum(a['delays'] for a in agents)/len(agents),2),
        'blocked_agents': len([a for a in agents if a.get('blocked_by',-1)!=-1]),
        'trend_counts':   trend_counts,
        'timestep':       50
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    print(f"XMAPF Flask app starting on port {port}...")
    app.run(debug=debug, host='0.0.0.0', port=port)
