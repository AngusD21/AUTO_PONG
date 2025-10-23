# MXEN3005 Robot Manipulation Project
# Forward and inverse kinematics code for TR WX250s
import numpy as np


# Inverse kinematics: input and ouput joints in degrees, input goal HTM in mm
def ik(thetas, goal_htm):
    # Joint angles
    thetas = list(thetas)

    # Newton params
    max_iters = 20
    step = 0.8

    converged = False
    for _ in range(max_iters):
        Fee, Jee = fk(thetas)
        residual = np.reshape(goal_htm - Fee, 16)
        htm_norm = np.sum(residual**2)
        if htm_norm < 1e-3:
            converged = True
            break

        dthetas = np.linalg.pinv(Jee) @ residual
        thetas += step * (180.0 / np.pi) * dthetas

    if converged:
        return thetas
    else:
        print("IK did not converge within 20 iterations")
        return None


# Forward kinematics and its Jacobian; input joints in degrees, output HTM in mm
def fk(thetas):
    dh, Tee = dh_from_joints(thetas)
    F1, F1der = htm_from_dh(dh[0])
    F2, F2der = htm_from_dh(dh[1])
    F3, F3der = htm_from_dh(dh[2])
    F4, F4der = htm_from_dh(dh[3])
    F5, F5der = htm_from_dh(dh[4])
    F6, F6der = htm_from_dh(dh[5])
    Fee = F1 @ F2 @ F3 @ F4 @ F5 @ F6 @ Tee

    Fee1 = np.reshape(F1der @ F2 @ F3 @ F4 @ F5 @ F6 @ Tee, 16)
    Fee2 = np.reshape(F1 @ F2der @ F3 @ F4 @ F5 @ F6 @ Tee, 16)
    Fee3 = np.reshape(F1 @ F2 @ F3der @ F4 @ F5 @ F6 @ Tee, 16)
    Fee4 = np.reshape(F1 @ F2 @ F3 @ F4der @ F5 @ F6 @ Tee, 16)
    Fee5 = np.reshape(F1 @ F2 @ F3 @ F4 @ F5der @ F6 @ Tee, 16)
    Fee6 = np.reshape(F1 @ F2 @ F3 @ F4 @ F5 @ F6der @ Tee, 16)

    Jee = np.stack((Fee1, Fee2, Fee3, Fee4, Fee5, Fee6), axis=-1)
    return Fee, Jee


# DH table and the transform from the last moving link to ee; input joints in degrees
def dh_from_joints(thetas):
    # Thetas
    th1, th2, th3, th4, th5, th6 = thetas
    # Link lengths in mm
    links = (110.25, 254.95, 171.55, 78.45, 65, 109.15)
    # Offset angle in degrees
    beta = 11.3
    # DH table
    dh = [
        [0, 0, links[0], th1],
        [0, 90, 0, 90 + th2 - beta],
        [links[1], 0, 0, th3 + beta],
        [0, 90, links[2] + links[3], th4],
        [0, -90, 0, th5],
        [0, 90, links[4], th6],
    ]
    Tee = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, links[5]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return dh, Tee


# HTM and its derivative wrt joint variable from DH table row
def htm_from_dh(dh_row):
    a = dh_row[0]
    al = np.pi / 180.0 * dh_row[1]
    d = dh_row[2]
    th = np.pi / 180.0 * dh_row[3]

    htm = np.array(
        [
            [np.cos(th), -np.sin(th), 0.0, a],
            [
                np.cos(al) * np.sin(th),
                np.cos(al) * np.cos(th),
                -np.sin(al),
                -np.sin(al) * d,
            ],
            [
                np.sin(al) * np.sin(th),
                np.sin(al) * np.cos(th),
                np.cos(al),
                np.cos(al) * d,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    htm_der = np.array(
        [
            [-np.sin(th), -np.cos(th), 0.0, 0.0],
            [
                np.cos(al) * np.cos(th),
                np.cos(al) * (-np.sin(th)),
                0.0,
                0.0,
            ],
            [
                np.sin(al) * np.cos(th),
                np.sin(al) * (-np.sin(th)),
                0.0,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    return htm, htm_der
