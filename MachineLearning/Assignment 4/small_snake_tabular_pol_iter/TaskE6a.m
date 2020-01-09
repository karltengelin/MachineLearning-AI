% Policy evaluation.
    while 1
        
        delta = 0;
        for state_idx = 1 : nbr_states
            s_prime = next_state_idxs(state_idx, policy(state_idx));
            V_s_prime = 0;                      %set to zero if we reach terminal state
        
            if s_prime == -1                    %as defined in get_states_next_state_idxs for eating an apple
                reward = rewards.apple;
            elseif s_prime == 0                 %as defined in get_states_next_state_idxs for moving into wall
                reward = rewards.death;
            else
                reward = rewards.default;       %means that we move into a new state without eating an apple or colliding
                V_s_prime = values(s_prime);    %future rewards, which we can only get if we dont't hit an apple or a wall
            end
            V_k1 = reward + gamm*V_s_prime;     %Bellman for V
            delta = abs(V_k1-values(state_idx));
            values(state_idx) = V_k1;           %We update the values (which corresponds to V^*)
        end
        
        % Increase nbr_pol_eval counter.
        nbr_pol_eval = nbr_pol_eval + 1;
        
        % Check for policy evaluation termination.
        if delta < pol_eval_tol
            break;
        else
            disp(['Delta: ', num2str(delta)])
        end
    end
    
    % Policy improvement.
    policy_stable = true; 
    for state_idx = 1 : nbr_states
        route = next_state_idxs(state_idx,:);   % the route the agent can take (i.e which states it can reach) given its actions
        old_policy = policy(state_idx);         % the old randomized policy 
        terminal = -1;                          % The smallest value V can have
        for i = 1:size(next_state_idxs,2)
            s_prime = route(i);
            V_s_prime = 0;
            
            if s_prime == -1
                reward = rewards.apple;
            
            elseif s_prime == 0
                reward = rewards.death;
            else
                reward = rewards.default;
                V_s_prime = values(s_prime);
            end
            if reward + gamm*V_s_prime > terminal
                policy(state_idx) = i;
                terminal = reward + gamm*V_s_prime;
            end
        end
        policy_stable = policy_stable & (policy(state_idx) == old_policy);
    end
    
    % Increase the number of policy iterations .
    nbr_pol_iter = nbr_pol_iter + 1;
    
    % Check for policy iteration termination (terminate if and only if the
    % policy is no longer changing, i.e. if and only if the policy is
    % stable).
    if policy_stable
        break;
    end