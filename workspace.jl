module workspace #inspired by ASEN 5254 AMP-Tools-Public package by Peter Amorese
    # Collin Hudson 11/16/2024
    using StaticArrays
    using Random
    using LazySets
    using JuMP, HiGHS
    export centroid
    export shrink
    export env
    export inLOS
    export inObs

    function inLOS(p1x,p1y,p2x,p2y,obs)
        noHit = true
        signal = LineSegment([p1x,p1y],[p2x,p2y])
            for ob in obs
                vs = ob.vertices
                for j in 1:(length(vs)-1)
                    if(noHit)
                        edge = LineSegment(vs[j],vs[j+1])
                        noHit = isdisjoint(signal, edge)
                    else
                        return false
                    end
                end
                if(noHit)
                    edge = LineSegment(vs[end],vs[1])
                    noHit = isdisjoint(signal, edge)
                else
                    return false
                end
            end
        return true
    end

    function inObs(p1x,p1y,obs)
        for ob in obs
            if ∈([p1x,p1y],ob)
                return true
            end
        end
        return false
    end

    function centroid(v::VPolygon)
        vts = [v.vertices;[v.vertices[1]]]
        centroid = [0.0;0.0]
        for j in 1:(length(vts)-1)
            centroid[1] += (vts[j][1] + vts[j+1][1])*(vts[j][1]*vts[j+1][2] - vts[j+1][1]*vts[j][2])
            centroid[2] += (vts[j][2] + vts[j+1][2])*(vts[j][1]*vts[j+1][2] - vts[j+1][1]*vts[j][2])
        end
        centroid = centroid/(6*area(v))
    end

    function shrink!(v::VPolygon,ρ::Float64)
        c = centroid(v)
        for j in 1:length(v.vertices)
            v.vertices[j]  = v.vertices[j] + ρ*(c - v.vertices[j])
        end
        return v
    end

    function shrink(v::VPolygon,ρ::Float64)
        v2 = copy(v)
        c = centroid(v2)
        for j in 1:length(v2.vertices)
            v2.vertices[j]  = v2.vertices[j] + ρ*(c - v2.vertices[j])
        end
        return v2
    end

    function checkArea(tempVec,xlim,ylim)
        tmpHull = convex_hull(tempVec[1],tempVec[2])
        for j = 3:length(tempVec)
            tmpHull = convex_hull(tmpHull,tempVec[j])
        end
        areaPoly = area(tmpHull)
        areaBounds = (xlim[2] - xlim[1])*(ylim[2] - ylim[1])
        if areaPoly > areaBounds
            return -1
        elseif areaPoly < 0.5*areaBounds
            return 1
        else
            return 0
        end
    end

    function scalePoly(v::VPolygon,xlim,ylim,)
        if diameter(v) > 0.75*min(xlim[2] - xlim[1],ylim[2] - ylim[1])
            return shrink!(v,0.5)
        else
            return v
        end
    end

    function polyMove(v::VPolygon,outVal)
            ang = -pi + 2*pi*rand()
            dirVec = [cos(ang);sin(ang)]
            return LazySets.translate(v,outVal*rand()*dirVec)
    end

    struct env
        xlim::SVector{2, Float64} #[min, max] limits for x coords
        ylim::SVector{2, Float64} #[min, max] limits for y coords
        obs #static vector of obstacles of type VPolygon (setting type caused bugs for some reason)
        Mobs #Vector of optimal M values
        Nverts #number of obstacle vertices
        function env(xlim,ylim,Nobs=5,seed=nothing)
            @assert xlim[2] > xlim[1] "Must have valid x bounds!"
            @assert ylim[2] > ylim[1] "Must have valid y bounds!"
            @assert Nobs >= 0 "Must have a non-negative number of obstacles!"
            if isnothing(seed)
                Random.seed!()
            else
                Random.seed!(seed)
            end
            #generate N obstacles with a random number of vertices (max 10)
            tempVec = []
            if Nobs > 0
                for j in 1:Nobs
                    push!(tempVec,LazySets.rand(VPolygon,num_vertices=rand(3:6)))
                end
                #Move all obstacles to be centered at origin of environment
                ctr = [sum(xlim)/2;sum(ylim)/2]
                tempVec = map(x->LazySets.translate(x,ctr-centroid(x)),tempVec)
                # Adjust size of polygons to fit bounds
                tempVec = map(x->scalePoly(x,xlim,ylim),tempVec)
                # Move polygons away from origin until footprint is large (or small) enough
                tooBig = checkArea(tempVec,xlim,ylim)
                while(tooBig != 0)
                    # Move polygons from environment origin a random distance in a random direction
                    tempVec = map(x->polyMove(x,tooBig),tempVec)
                    tooBig = checkArea(tempVec,xlim,ylim)
                end
                obsVec = SVector{Nobs,VPolygon}(tempVec)
                # Solve for optimal M values for obstacle avoidance
                Nverts = sum(x->length(x.vertices),obsVec)
                ks = zeros(Nverts,1)
                hs = zeros(Nverts,2*Nverts)
                row = 1
                for ob in obsVec
                    for h in LazySets.constraints_list(ob)
                        hs[row,:] = hcat(zeros(1,2*(row-1)),h.a',zeros(1,2*(Nverts - row)))
                        ks[row] = h.b
                        row += 1
                    end
                end
                model = Model(HiGHS.Optimizer)
                set_silent(model)
                @variable(model, x[i=1:2*Nverts])
                for j in 1:2:2*Nverts
                    @constraint(model,xlim[1] <= x[j] <= xlim[2])
                    @constraint(model,ylim[1] <= x[j+1] <= ylim[2])
                end
                @objective(model, Max, sum(ks - hs*x))
                optimize!(model)
                # Mobs = maximum(ks - hs*value.(x))*ones(length(value.(x)),1)
                Mobs = ks - hs*value.(x)
            else
                obsVec = SVector{Nobs,VPolygon}([])
                Mobs = []
                Nverts = 0
            end
            new(xlim,ylim,obsVec,Mobs,Nverts)
        end
    end

end