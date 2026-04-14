using Test, LinearAlgebra
using DualArrays: Tensor

@testset "Tensor" begin
    t = Tensor{1}([1 2 3;4 5 6;7 8 9])
    @testset "Indexing" begin
        @test t[1,1] == 1
        @test t[(1,), (:,)] == Tensor{0}([1, 2, 3])
        @test t[1:2, 1:2] == Tensor{1}([1 2;4 5])

        s = Tensor{2}(ones(2,2,2))
        @test s[(1,1), (:,)] == Tensor{0}([1, 1])
        @test s[(1,:), (:,)] == Tensor{1}([1 1;1 1])
        @test s[(1,2), :] == Tensor{0}([1, 1])
    end

    @testset "Arithmetic" begin
        @test t + t isa Tensor
        @test t + t == Tensor{1}([2 4 6;8 10 12;14 16 18])

        @test t .* 2 isa Tensor
        @test t .* 2 == Tensor{1}([2 4 6;8 10 12;14 16 18])
        @test t .* [1,2,3] == Tensor{1}([1 2 3;8 10 12;21 24 27])

        t1 = Tensor{0}([1, 2, 3])
        t2 = Tensor{1}([1 2 3])
        t3 = Tensor{0}([2])

        @test t1 .* t2 == Tensor{1}([1 2 3;2 4 6;3 6 9])
        @test t1 .* t2 .+ t3 == Tensor{1}([3 4 5;4 6 8;5 8 11])

        @test sin.(t1) == Tensor{0}(sin.([1, 2, 3]))

        s = Tensor{1}(zeros(3, 3))

        result = (s .= t .+ 1)

        @test result isa Tensor
        @test result.data == s.data
        @test s == Tensor{1}([2 3 4; 5 6 7; 8 9 10])
    end
end