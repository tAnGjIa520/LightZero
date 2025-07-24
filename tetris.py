#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import time
import sys
import os
import select
import tty
import termios

class Tetris:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = [[0 for _ in range(width)] for _ in range(height)]
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.fall_time = 0.5
        self.game_over = False
        
        # 定义俄罗斯方块的形状
        self.shapes = {
            'I': [
                ['.....',
                 '..#..',
                 '..#..',
                 '..#..',
                 '..#..'],
                ['.....',
                 '.....',
                 '####.',
                 '.....',
                 '.....']
            ],
            'O': [
                ['.....',
                 '.....',
                 '.##..',
                 '.##..',
                 '.....']
            ],
            'T': [
                ['.....',
                 '.....',
                 '.#...',
                 '###..',
                 '.....'],
                ['.....',
                 '.....',
                 '.#...',
                 '.##..',
                 '.#...'],
                ['.....',
                 '.....',
                 '.....',
                 '###..',
                 '.#...'],
                ['.....',
                 '.....',
                 '.#...',
                 '##...',
                 '.#...']
            ],
            'S': [
                ['.....',
                 '.....',
                 '.##..',
                 '##...',
                 '.....'],
                ['.....',
                 '.#...',
                 '.##..',
                 '..#..',
                 '.....']
            ],
            'Z': [
                ['.....',
                 '.....',
                 '##...',
                 '.##..',
                 '.....'],
                ['.....',
                 '..#..',
                 '.##..',
                 '.#...',
                 '.....']
            ],
            'J': [
                ['.....',
                 '.#...',
                 '.#...',
                 '##...',
                 '.....'],
                ['.....',
                 '.....',
                 '#....',
                 '###..',
                 '.....'],
                ['.....',
                 '.##..',
                 '.#...',
                 '.#...',
                 '.....'],
                ['.....',
                 '.....',
                 '###..',
                 '..#..',
                 '.....']
            ],
            'L': [
                ['.....',
                 '..#..',
                 '..#..',
                 '.##..',
                 '.....'],
                ['.....',
                 '.....',
                 '###..',
                 '#....',
                 '.....'],
                ['.....',
                 '##...',
                 '.#...',
                 '.#...',
                 '.....'],
                ['.....',
                 '.....',
                 '..#..',
                 '###..',
                 '.....']
            ]
        }
        
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        
    def new_piece(self):
        """创建新的方块"""
        shape_name = random.choice(list(self.shapes.keys()))
        return {
            'shape': shape_name,
            'rotation': 0,
            'x': self.width // 2 - 2,
            'y': 0,
            'color': ord(shape_name)
        }
    
    def get_shape_cells(self, piece):
        """获取方块的所有单元格位置"""
        shape = self.shapes[piece['shape']][piece['rotation']]
        cells = []
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell == '#':
                    cells.append((piece['y'] + i, piece['x'] + j))
        return cells
    
    def is_valid_position(self, piece):
        """检查方块位置是否有效"""
        cells = self.get_shape_cells(piece)
        for y, x in cells:
            if x < 0 or x >= self.width or y >= self.height:
                return False
            if y >= 0 and self.board[y][x] != 0:
                return False
        return True
    
    def place_piece(self, piece):
        """将方块放置到游戏板上"""
        cells = self.get_shape_cells(piece)
        for y, x in cells:
            if y >= 0:
                self.board[y][x] = piece['color']
    
    def clear_lines(self):
        """清除满行"""
        lines_to_clear = []
        for y in range(self.height):
            if all(self.board[y][x] != 0 for x in range(self.width)):
                lines_to_clear.append(y)
        
        for y in sorted(lines_to_clear, reverse=True):
            del self.board[y]
            self.board.insert(0, [0 for _ in range(self.width)])
        
        lines_cleared = len(lines_to_clear)
        if lines_cleared > 0:
            self.lines_cleared += lines_cleared
            self.score += lines_cleared * 100 * self.level
            self.level = min(10, 1 + self.lines_cleared // 10)
            self.fall_time = max(0.1, 0.5 - (self.level - 1) * 0.05)
    
    def move_piece(self, dx, dy, dr=0):
        """移动方块"""
        new_piece = self.current_piece.copy()
        new_piece['x'] += dx
        new_piece['y'] += dy
        new_piece['rotation'] = (new_piece['rotation'] + dr) % len(self.shapes[new_piece['shape']])
        
        if self.is_valid_position(new_piece):
            self.current_piece = new_piece
            return True
        return False
    
    def drop_piece(self):
        """方块自然下落"""
        if not self.move_piece(0, 1):
            # 无法继续下落，放置方块
            self.place_piece(self.current_piece)
            self.clear_lines()
            
            # 生成新方块
            self.current_piece = self.next_piece
            self.next_piece = self.new_piece()
            
            # 检查游戏是否结束
            if not self.is_valid_position(self.current_piece):
                self.game_over = True
    
    def hard_drop(self):
        """硬降（直接落到底部）"""
        while self.move_piece(0, 1):
            pass
        self.drop_piece()
    
    def draw_board(self):
        """绘制游戏界面"""
        # 清屏
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # 创建显示板
        display_board = [row[:] for row in self.board]
        
        # 绘制当前方块
        if not self.game_over:
            cells = self.get_shape_cells(self.current_piece)
            for y, x in cells:
                if 0 <= y < self.height and 0 <= x < self.width:
                    display_board[y][x] = self.current_piece['color']
        
        # 打印游戏标题和信息
        print("=" * 50)
        print(f"{'俄罗斯方块':^48}")
        print("=" * 50)
        print(f"分数: {self.score:<10} 等级: {self.level:<5} 已清除行数: {self.lines_cleared}")
        print("-" * 50)
        
        # 打印游戏区域
        print("┌" + "─" * (self.width * 2) + "┐")
        for row in display_board:
            line = "│"
            for cell in row:
                if cell == 0:
                    line += "  "
                else:
                    line += "██"
            line += "│"
            print(line)
        print("└" + "─" * (self.width * 2) + "┘")
        
        # 打印下一个方块预览
        print(f"\n下一个方块:")
        if self.next_piece:
            shape = self.shapes[self.next_piece['shape']][0]
            for row in shape:
                preview_line = ""
                for cell in row:
                    if cell == '#':
                        preview_line += "██"
                    else:
                        preview_line += "  "
                if preview_line.strip():
                    print(preview_line)
        
        # 打印控制说明
        print(f"\n控制说明:")
        print("A/D - 左右移动  S - 向下移动  W - 旋转")
        print("空格 - 硬降  Q - 退出游戏")
        
        if self.game_over:
            print(f"\n{'游戏结束！':^20}")
            print(f"最终分数: {self.score}")
    
    def get_input(self):
        """获取用户输入（非阻塞）"""
        if select.select([sys.stdin], [], [], 0.1)[0]:
            return sys.stdin.read(1).lower()
        return None
    
    def run(self):
        """运行游戏主循环"""
        # 设置终端为非阻塞模式
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.cbreak(sys.stdin)
            
            last_fall_time = time.time()
            
            while not self.game_over:
                self.draw_board()
                
                # 处理输入
                key = self.get_input()
                if key:
                    if key == 'q':
                        break
                    elif key == 'a':
                        self.move_piece(-1, 0)
                    elif key == 'd':
                        self.move_piece(1, 0)
                    elif key == 's':
                        self.move_piece(0, 1)
                    elif key == 'w':
                        self.move_piece(0, 0, 1)
                    elif key == ' ':
                        self.hard_drop()
                
                # 自然下落
                current_time = time.time()
                if current_time - last_fall_time > self.fall_time:
                    self.drop_piece()
                    last_fall_time = current_time
                
                time.sleep(0.05)  # 控制游戏刷新率
            
            # 游戏结束显示
            self.draw_board()
            input("\n按回车键退出...")
            
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def main():
    """主函数"""
    print("欢迎来到俄罗斯方块！")
    print("准备开始游戏...")
    time.sleep(2)
    
    game = Tetris()
    game.run()

if __name__ == "__main__":
    main()
